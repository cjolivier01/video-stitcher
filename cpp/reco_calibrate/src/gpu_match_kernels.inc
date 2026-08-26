constexpr char kGpuMatchKernelSource[] = R"cuda(
// Runtime-compiled matching kernels. All pointer arguments refer to device memory.
// match_scan_block must be launched with exactly 256 threads per block. Its block
// sums may be scanned recursively; match_scan_add then propagates each scanned
// block prefix toward the preceding level. Input and output of match_stable_sort
// must not alias.

struct GpuMatchFeaturePoint {
  float x;
  float y;
  float response;
  float size;
  unsigned int octave;
  unsigned int level;
};

struct GpuCompactMatch {
  float left_x;
  float left_y;
  float left_response;
  float right_x;
  float right_y;
  float right_response;
  unsigned int distance;
};

static_assert(sizeof(GpuMatchFeaturePoint) == 24, "feature point ABI must be 24 bytes");
static_assert(sizeof(GpuCompactMatch) == 28, "compact match ABI must be 28 bytes");

namespace {

constexpr unsigned int kInvalidMatchIndex = 0xffffffffU;
constexpr unsigned int kInvalidMatchDistance = 0xffffffffU;
constexpr unsigned int kDescriptorBytes = 64U;
constexpr unsigned int kDescriptorWords = 8U;
constexpr unsigned int kScanBlockSize = 256U;

__device__ __forceinline__ unsigned int bounded_count(const unsigned int* count,
                                                       unsigned int capacity) {
  const unsigned int value = *count;
  return value < capacity ? value : capacity;
}

__device__ __forceinline__ unsigned int descriptor_distance(
    const unsigned char* query,
    const unsigned char* train) {
  const unsigned long long* query_words =
      reinterpret_cast<const unsigned long long*>(query);
  const unsigned long long* train_words =
      reinterpret_cast<const unsigned long long*>(train);
  unsigned int distance = 0U;
#pragma unroll
  for (unsigned int word = 0U; word < kDescriptorWords; ++word) {
    distance += __popcll(query_words[word] ^ train_words[word]);
  }
  return distance;
}

} // namespace

// Launch once in each direction. One thread owns one query descriptor, so tie
// handling and the best/second-best reductions are independent of scheduling.
extern "C" __global__ void match_one_way(
    const unsigned char* query_descriptors,
    const unsigned int* query_count,
    unsigned int query_capacity,
    const unsigned char* train_descriptors,
    const unsigned int* train_count,
    unsigned int train_capacity,
    float lowe_ratio,
    unsigned int* best_indices,
    unsigned int* best_distances,
    unsigned int* second_distances,
    unsigned int* accepted) {
  const unsigned int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= query_capacity) {
    return;
  }

  const unsigned int queries = bounded_count(query_count, query_capacity);
  const unsigned int trains = bounded_count(train_count, train_capacity);
  if (query_index >= queries) {
    best_indices[query_index] = kInvalidMatchIndex;
    best_distances[query_index] = kInvalidMatchDistance;
    second_distances[query_index] = kInvalidMatchDistance;
    accepted[query_index] = 0U;
    return;
  }

  const unsigned char* query =
      query_descriptors + static_cast<unsigned long long>(query_index) * kDescriptorBytes;
  unsigned int best_index = kInvalidMatchIndex;
  unsigned int best_distance = kInvalidMatchDistance;
  unsigned int second_distance = kInvalidMatchDistance;

  for (unsigned int train_index = 0U; train_index < trains; ++train_index) {
    const unsigned char* train =
        train_descriptors + static_cast<unsigned long long>(train_index) * kDescriptorBytes;
    const unsigned int distance = descriptor_distance(query, train);
    if (distance < best_distance) {
      second_distance = best_distance;
      best_distance = distance;
      best_index = train_index;
    } else if (distance < second_distance) {
      // An equal best distance becomes the second best. The first (lowest)
      // train index remains the winner.
      second_distance = distance;
    }
  }

  unsigned int passes = 0U;
  if (lowe_ratio < 1.0F) {
    passes = second_distance > 0U &&
                     static_cast<float>(best_distance) <
                         lowe_ratio * static_cast<float>(second_distance)
                 ? 1U
                 : 0U;
  } else {
    // The validated public API permits exactly 1.0 as the no-ratio-test case.
    passes = best_index != kInvalidMatchIndex ? 1U : 0U;
  }

  best_indices[query_index] = best_index;
  best_distances[query_index] = best_distance;
  second_distances[query_index] = second_distance;
  accepted[query_index] = passes;
}

// Converts independently accepted left-to-right and right-to-left winners into
// scan-ready zero/one flags. Launch across left_capacity, including unused slots.
extern "C" __global__ void match_crosscheck_flags(
    const unsigned int* left_best_indices,
    const unsigned int* left_accepted,
    const unsigned int* right_best_indices,
    const unsigned int* right_accepted,
    const unsigned int* left_count,
    unsigned int left_capacity,
    const unsigned int* right_count,
    unsigned int right_capacity,
    unsigned int* flags) {
  const unsigned int left_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (left_index >= left_capacity) {
    return;
  }

  const unsigned int left_size = bounded_count(left_count, left_capacity);
  const unsigned int right_size = bounded_count(right_count, right_capacity);
  unsigned int keep = 0U;
  if (left_index < left_size && left_accepted[left_index] != 0U) {
    const unsigned int right_index = left_best_indices[left_index];
    if (right_index < right_size && right_accepted[right_index] != 0U &&
        right_best_indices[right_index] == left_index) {
      keep = 1U;
    }
  }
  flags[left_index] = keep;
}

// Deterministic exclusive scan of one level. element_count is a host-known
// allocation extent, not a feature count readback. Scan block_sums recursively
// when gridDim.x is greater than one.
extern "C" __global__ void match_scan_block(
    const unsigned int* input,
    unsigned int* offsets,
    unsigned int* block_sums,
    unsigned int element_count) {
  __shared__ unsigned int values[kScanBlockSize];
  if (blockDim.x != kScanBlockSize) {
    return;
  }

  const unsigned int lane = threadIdx.x;
  const unsigned int index = blockIdx.x * kScanBlockSize + lane;
  const unsigned int value = index < element_count ? input[index] : 0U;
  values[lane] = value;
  __syncthreads();

  for (unsigned int stride = 1U; stride < kScanBlockSize; stride <<= 1U) {
    const unsigned int addend = lane >= stride ? values[lane - stride] : 0U;
    __syncthreads();
    values[lane] += addend;
    __syncthreads();
  }

  if (index < element_count) {
    offsets[index] = values[lane] - value;
  }
  if (lane == kScanBlockSize - 1U) {
    block_sums[blockIdx.x] = values[lane];
  }
}

// Adds recursively scanned block prefixes to a lower scan level. The launch
// block size is unrestricted; scan_block_size describes the lower level and is
// normally 256.
extern "C" __global__ void match_scan_add(
    unsigned int* offsets,
    const unsigned int* block_offsets,
    unsigned int element_count,
    unsigned int scan_block_size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count || scan_block_size == 0U) {
    return;
  }
  offsets[index] += block_offsets[index / scan_block_size];
}

// Compaction is stable because exclusive offsets are assigned in ascending
// left-index order. Thread zero also publishes the compact count for the next
// launch without a count readback.
extern "C" __global__ void match_scatter(
    const GpuMatchFeaturePoint* left_points,
    const unsigned int* left_count,
    unsigned int left_capacity,
    const GpuMatchFeaturePoint* right_points,
    const unsigned int* right_count,
    unsigned int right_capacity,
    const unsigned int* left_best_indices,
    const unsigned int* left_best_distances,
    const unsigned int* flags,
    const unsigned int* offsets,
    GpuCompactMatch* compact_matches,
    unsigned int compact_capacity,
    unsigned int* compact_count) {
  const unsigned int left_index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int left_size = bounded_count(left_count, left_capacity);
  const unsigned int right_size = bounded_count(right_count, right_capacity);

  if (left_index == 0U) {
    const unsigned int total = left_capacity == 0U
                                   ? 0U
                                   : offsets[left_capacity - 1U] + flags[left_capacity - 1U];
    *compact_count = total < compact_capacity ? total : compact_capacity;
  }
  if (left_index >= left_size || flags[left_index] == 0U) {
    return;
  }

  const unsigned int output_index = offsets[left_index];
  const unsigned int right_index = left_best_indices[left_index];
  if (output_index >= compact_capacity || right_index >= right_size) {
    return;
  }

  const GpuMatchFeaturePoint left = left_points[left_index];
  const GpuMatchFeaturePoint right = right_points[right_index];
  GpuCompactMatch result;
  result.left_x = left.x;
  result.left_y = left.y;
  result.left_response = left.response;
  result.right_x = right.x;
  result.right_y = right.y;
  result.right_response = right.response;
  result.distance = left_best_distances[left_index];
  compact_matches[output_index] = result;
}

// Scatter has already ordered records by left index. Ranking by distance and
// then by compact input index therefore yields a stable (distance, left index)
// order with one unique writer per output slot and no atomic append.
extern "C" __global__ void match_stable_sort(
    const GpuCompactMatch* compact_matches,
    const unsigned int* compact_count,
    unsigned int compact_capacity,
    GpuCompactMatch* sorted_matches) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int count = bounded_count(compact_count, compact_capacity);
  if (index >= count) {
    return;
  }

  const unsigned int distance = compact_matches[index].distance;
  unsigned int rank = 0U;
  for (unsigned int other = 0U; other < count; ++other) {
    const unsigned int other_distance = compact_matches[other].distance;
    if (other_distance < distance || (other_distance == distance && other < index)) {
      ++rank;
    }
  }
  sorted_matches[rank] = compact_matches[index];
}
)cuda";

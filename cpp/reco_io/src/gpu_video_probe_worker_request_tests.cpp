#include "gpu_video_probe_worker_request.hpp"

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using reco::io::detail::ReceivedRequest;

static_assert(!std::is_copy_constructible_v<ReceivedRequest>);
static_assert(!std::is_copy_assignable_v<ReceivedRequest>);
static_assert(std::is_nothrow_move_constructible_v<ReceivedRequest>);
static_assert(std::is_nothrow_move_assignable_v<ReceivedRequest>);

int failures = 0;

void fail(const std::string& message) {
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

#if defined(_WIN32)
void move_preserves_payload() {
  ReceivedRequest source("request");
  ReceivedRequest destination(std::move(source));
  if (destination.payload() != "request") {
    fail("move construction preserves the request payload");
  }

  ReceivedRequest assigned("old");
  assigned = std::move(destination);
  if (assigned.payload() != "request") {
    fail("move assignment preserves the request payload");
  }
}
#else
bool is_open(int descriptor) {
  errno = 0;
  return ::fcntl(descriptor, F_GETFD) >= 0 || errno != EBADF;
}

bool open_pipe(int (&descriptors)[2], const std::string& message) {
  if (::pipe(descriptors) == 0) {
    return true;
  }
  fail(message);
  return false;
}

void move_construction_transfers_ownership() {
  int pipe_descriptors[2];
  if (!open_pipe(pipe_descriptors, "create pipe for move construction")) {
    return;
  }
  const int owned_descriptor = pipe_descriptors[0];
  {
    ReceivedRequest source("request", owned_descriptor);
    ReceivedRequest destination(std::move(source));
    if (source.descriptor() != -1) {
      fail("move construction clears the source descriptor");
    }
    if (destination.descriptor() != owned_descriptor || !is_open(owned_descriptor)) {
      fail("move construction transfers the open descriptor");
    }
    if (destination.payload() != "request") {
      fail("move construction preserves the request payload");
    }
  }
  if (is_open(owned_descriptor)) {
    fail("destroying a moved-to request closes its descriptor");
    (void)::close(owned_descriptor);
  }
  (void)::close(pipe_descriptors[1]);
}

void move_assignment_replaces_ownership() {
  int destination_pipe[2];
  if (!open_pipe(destination_pipe, "create destination pipe for move assignment")) {
    return;
  }
  int source_pipe[2];
  if (!open_pipe(source_pipe, "create source pipe for move assignment")) {
    (void)::close(destination_pipe[0]);
    (void)::close(destination_pipe[1]);
    return;
  }
  const int replaced_descriptor = destination_pipe[0];
  const int transferred_descriptor = source_pipe[0];
  {
    ReceivedRequest destination("old", replaced_descriptor);
    ReceivedRequest source("new", transferred_descriptor);
    destination = std::move(source);
    if (is_open(replaced_descriptor)) {
      fail("move assignment closes the destination's previous descriptor");
    }
    if (source.descriptor() != -1) {
      fail("move assignment clears the source descriptor");
    }
    if (destination.descriptor() != transferred_descriptor || !is_open(transferred_descriptor)) {
      fail("move assignment transfers the source descriptor");
    }
    if (destination.payload() != "new") {
      fail("move assignment preserves the source payload");
    }
  }
  if (is_open(transferred_descriptor)) {
    fail("destroying a move-assigned request closes its descriptor");
    (void)::close(transferred_descriptor);
  }
  (void)::close(destination_pipe[1]);
  (void)::close(source_pipe[1]);
}

void released_descriptor_remains_open() {
  int pipe_descriptors[2];
  if (!open_pipe(pipe_descriptors, "create pipe for descriptor release")) {
    return;
  }
  const int owned_descriptor = pipe_descriptors[0];
  {
    ReceivedRequest request("request", owned_descriptor);
    if (request.release_descriptor() != owned_descriptor || request.descriptor() != -1) {
      fail("release returns the descriptor and clears request ownership");
    }
  }
  if (!is_open(owned_descriptor)) {
    fail("destroying a request does not close its released descriptor");
  }
  (void)::close(owned_descriptor);
  (void)::close(pipe_descriptors[1]);
}
#endif

} // namespace

int main() {
#if defined(_WIN32)
  move_preserves_payload();
#else
  move_construction_transfers_ownership();
  move_assignment_replaces_ownership();
  released_descriptor_remains_open();
#endif
  return failures == 0 ? 0 : 1;
}

# GPU preview controller friction

- `GpuFileDecodeConfig` cannot select an NVDEC/DeepStream GPU. The controller
  must reject nonzero device ordinals until `reco_io` can bind decode, NVMM
  mapping, CUDA rendering, and presentation to the same configured device.
- `open_gstreamer_gpu_file_decode_source` has no cancellation input and returns
  only after native pipeline setup and an optional indexed seek. Normal blocked
  reads are interruptible, but stop or a newer seek cannot preempt a decoder
  rebuild until that bounded native setup call returns.

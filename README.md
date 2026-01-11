## Int8 Quantized Inference Engine

A custom C++/CUDA extension for PyTorch that performs high-speed matrix multiplication (GEMM) using Int8 precision.

Core implementation details:
* __Tensor Cores__: Uses NVIDIA WMMA API to execute matrix math on A100 GPUs
* __Memory Optimization__: Implements shared memory tiling to minimize high-latency VRAM access
* __Cooperative Loading__: Uses multi-threaded cooperative fetching to ensure coalesced memory reads
* __Performance__: Achieves 33x speedup from PyTorch CPU baselines

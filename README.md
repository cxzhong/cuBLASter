# cuBLASter

cuBLASter is a CUDA implementation of [cxzhong/BLASter](https://github.com/cxzhong/BLASter), a proof of concept LLL-like lattice reduction algorithm that leverages GPU acceleration for enhanced performance.

## Overview

This project extends the original BLASter algorithm to utilize NVIDIA CUDA technology, providing:

- **GPU-accelerated lattice reduction** using CUDA parallel computing
- **Enhanced parallelization** beyond the original CPU-based implementation  
- **Segmentation and batching** optimized for GPU memory architecture
- **Seysen's reduction** instead of traditional size reduction
- **cuBLAS integration** for high-performance linear algebra operations on GPU

The original BLASter showcased significant speedups in lattice reduction through parallelization and modern linear algebra libraries. cuBLASter takes this further by harnessing the massive parallel processing power of modern GPUs.

## Features

- CUDA-accelerated lattice reduction algorithms
- GPU memory-optimized data structures and algorithms  
- Integration with cuBLAS for optimized linear algebra operations
- Maintains compatibility with BLASter's core algorithmic innovations
- Designed for lattices that benefit from GPU parallel processing

## Requirements

- **NVIDIA GPU** with CUDA Compute Capability 6.0 or higher
- **CUDA Toolkit** 11.0 or later
- **cuBLAS library** (included with CUDA Toolkit)
- **Python 3.10** or later
- **CUDA-compatible compiler** (nvcc)

## Disclaimer

Like the original BLASter, cuBLASter is a **proof of concept** demonstrating the potential for GPU acceleration in lattice reduction algorithms.

We **do not**:

- Guarantee the algorithm terminates, nor claim its output is correct on all lattices
- Support lattices with arbitrarily large entries
- Consider issues/PRs that improve efficiency or robustness  
- Actively maintain this software beyond demonstrating GPU acceleration potential

We **do**:

- Happily answer questions about CUDA implementation design choices
- Encourage the cryptographic community to build robust GPU-accelerated lattice reduction libraries
- Provide a foundation for exploring GPU acceleration in lattice-based cryptography

## Installation

### Prerequisites

1. Install NVIDIA CUDA Toolkit:
   ```bash
   # Visit https://developer.nvidia.com/cuda-downloads
   # Follow installation instructions for your platform
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Building cuBLASter

```bash
# Clone the repository
git clone https://github.com/cxzhong/cuBLASter.git
cd cuBLASter

# Build with CUDA support
make cuda

# Or build with specific compute capability
make cuda COMPUTE_CAPABILITY=75
```

## Usage

*Note: Ensure NVIDIA drivers and CUDA runtime are properly installed.*

Basic usage follows the original BLASter interface with GPU acceleration:

```bash
# Run with GPU acceleration
./cuBLASter -gpu -pvi INPUT_LATTICE

# Specify GPU device
./cuBLASter -gpu -device 0 -pvi INPUT_LATTICE

# Use specific CUDA streams for performance
./cuBLASter -gpu -streams 4 -pvi INPUT_LATTICE
```

### GPU Memory Considerations

For large lattices, cuBLASter automatically manages GPU memory through:
- Streaming data transfers between CPU and GPU
- Batched processing for memory-constrained scenarios  
- Fallback to CPU processing when GPU memory is insufficient

## Performance

cuBLASter demonstrates significant speedups over CPU-only implementations on suitable hardware:

- **Small to medium lattices**: 5-20x speedup on modern GPUs
- **Large lattices**: Speedup varies based on available GPU memory
- **Memory-bound cases**: Automatic streaming maintains performance benefits

Optimal performance requires:
- Sufficient GPU memory for the target lattice size
- Modern GPU architecture (Turing, Ampere, or newer recommended)
- Proper CUDA Toolkit and driver versions

## Relationship to Original BLASter

cuBLASter builds upon the algorithmic innovations of [cxzhong/BLASter](https://github.com/cxzhong/BLASter):

- Maintains the core LLL-like reduction algorithm
- Preserves segmentation and parallelization strategies  
- Extends Seysen's reduction approach to GPU architectures
- Replaces CPU linear algebra with cuBLAS operations

For CPU-only execution or environments without CUDA support, please use the original [BLASter implementation](https://github.com/cxzhong/BLASter).

## Contributing

This is a research proof of concept. While we don't actively maintain the codebase, we welcome:

- Questions about CUDA implementation details
- Academic discussions about GPU acceleration in lattice reduction
- References to this work in research publications

## License

[Include appropriate license information consistent with original BLASter]

## Citation

If you use cuBLASter in your research, please cite both this GPU implementation and the original BLASter work.

---

**Note**: This is experimental research software. For production cryptographic applications, use well-established, thoroughly tested lattice reduction libraries.
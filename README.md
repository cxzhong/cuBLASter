# cuBLASter

cuBLASter is a CUDA-accelerated version of BLASter, a proof of concept of an LLL-like lattice reduction algorithm that uses:

- **CUDA/cuBLAS** for GPU-accelerated linear algebra operations,
- parallelization on GPU,
- segmentation,
- Seysen's reduction instead of size reduction, and
- GPU memory management.

This is the GPU counterpart to [BLASter](https://github.com/cxzhong/BLASter), leveraging NVIDIA GPUs for high-performance lattice reduction.

## Disclaimer

The goal of this software is to showcase speed ups that are possible in lattice reduction software using GPU acceleration.
This software is a *proof of concept*!

In particular, we **do not**:

- guarantee the algorithm terminates, nor claim its output is correct on all lattices,
- support lattices with large entries,
- consider issues / PRs that improve efficiency or robustness,
- actively maintain this software.

We **do**:

- happily answer any questions to explain design choices phrased as: *"Why is X done in Y way?"*. The answer may, in many cases, be: "because it is faster in practice on GPU".
- encourage the cryptographic community to build a new robust lattice reduction library incorporating the ideas in this proof of concept.

## Requirements

- python3 (3.10 or later)
- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
- **CUDA Toolkit** (11.0 or later recommended)
- **cuBLAS library** (included with CUDA Toolkit)
- Cython version 3.0 or later (installed automatically)
- Python modules: `cysignals numpy setuptools matplotlib` (installed automatically)

Optional:

- Python module: virtualenv (for creating a local virtual environment to install python3 modules)
- fplll (for generating q-ary lattices with the `latticegen` command)

## Installation

### CUDA Setup (Optional)

If you have an NVIDIA GPU and want GPU acceleration:

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# On Ubuntu/Debian, you can install CUDA with:
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
# sudo cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-3
```

### Build from Source

cuBLASter automatically detects CUDA availability and falls back to CPU implementation if CUDA is not available:

```bash
# Install build dependencies
pip install numpy cython cysignals setuptools wheel

# Build the extension
python setup.py build_ext --inplace

# Test the installation
python test_app.py
```

### Modern PEP 517 Installation (when network available)

```bash
# Install from source (automatically detects CUDA)
pip install -e .

# Or build wheel for distribution
python -m build
```

### Environment Variables (for CUDA)

You may need to set these environment variables if CUDA is installed in a non-standard location:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Running

### Basic Usage

```bash
# Build the extension
python setup.py build_ext --inplace

# Test the core functionality
python test_app.py

# Test matrix operations directly
python -c "
import sys; sys.path.insert(0, '.')
import cublaster_core
cublaster_core.init_cuda(0)
print('Memory:', cublaster_core.get_memory_info())
"
```

*Note: The full application interface (src/app.py) requires proper package installation.*

## Example

### Matrix Operations Test

```python
import cublaster_core
import numpy as np

# Initialize (automatically detects CUDA or uses CPU fallback)
cublaster_core.init_cuda(0)

# Test matrix multiplication
A = np.array([[1, 2], [3, 4]], dtype=np.int64)
B = np.array([[5, 6], [7, 8]], dtype=np.int64)
C = cublaster_core.matmul(A, B)
print("A @ B =", C)  # Output: [[19, 22], [43, 50]]

# Check memory usage
mem_info = cublaster_core.get_memory_info()
print(f"Memory: {mem_info['used_mb']}MB used / {mem_info['total_mb']}MB total")
```

### Command Line Test

```bash
python test_app.py
```

Expected output:
```
cuBLASter Application Test
==================================================
Warning: CUDA not available, using CPU fallback  # (or GPU info if CUDA available)
GPU Memory: 0MB used / 1024MB total
Lattice dimensions: 4 x 4
Input lattice:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

Self-multiplication result:
[[ 90 100 110 120]
 [202 228 254 280]
 [314 356 398 440]
 [426 484 542 600]]

Profile: [0.74, 1.12, 1.32, 1.46]
∥b_1∥ = 5.477

✓ Application test completed successfully
```

## GPU Memory Management

cuBLASter automatically manages GPU memory and will display memory usage information. For large lattices, ensure your GPU has sufficient memory.

## Current Implementation Status

This implementation provides:

✅ **Core Infrastructure**: Complete build system with CUDA and CPU fallback  
✅ **Matrix Operations**: GPU-accelerated matrix multiplication using cuBLAS  
✅ **Memory Management**: Automatic GPU memory allocation and cleanup  
✅ **Python Bindings**: Cython-based interface with proper error handling  
✅ **CPU Fallback**: Automatic fallback when CUDA is not available  

🚧 **Lattice Reduction Algorithms**: Currently implemented as placeholder  
   - The core lattice reduction algorithms from BLASter need to be ported
   - Matrix operations are working and ready for algorithm implementation
   - Structure is in place for deep-LLL, BKZ, and other algorithms

🔧 **Coming Next**: 
   - Port core lattice reduction algorithms to use GPU matrix operations
   - Implement progressive BKZ with GPU acceleration
   - Add comprehensive test suite
   - Performance benchmarking against CPU version
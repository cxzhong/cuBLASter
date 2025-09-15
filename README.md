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

### CUDA Setup

First, ensure you have CUDA Toolkit installed:

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

### Modern PEP 517 Installation (Recommended)

cuBLASter supports modern Python packaging with automatic dependency management:

```bash
# Install from source (automatically detects CUDA)
pip install -e .

# Or build wheel for distribution
python -m build
```

### Environment Variables

You may need to set these environment variables:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Running

*Note: you first need to build the software, see above.*

Run the command by e.g. typing `./python3 src/app.py -pvi INPUTFILE`.
Add `-h` for seeing all available command line arguments.

## Example

Command: `time latticegen q 128 64 20 p | src/app.py -pq`.

Expected output:
```
Profile: [9.38 9.39 9.30 9.28 9.18 ... 4.27 4.33 4.31 4.29 4.35]
Root Hermite factor: 1.020447, ∥b_1∥ = 11906.636
GPU Memory Used: 245MB

real	0m0,354s
user	0m1,271s
sys	0m0,095s
```

To run deep-LLL with depth `4`, run `src/app.py -pq -i {lattice} -d4`.

To run progressive BKZ-60 (with 4-deep-LLL) with `1` tours and block size increments of `2`, run `src/app.py -pq -i {lattice} -b60 -t1 -P2`.

## GPU Memory Management

cuBLASter automatically manages GPU memory and will display memory usage information. For large lattices, ensure your GPU has sufficient memory.
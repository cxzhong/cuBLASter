"""
cuBLASter: CUDA-accelerated lattice reduction library

This package provides GPU-accelerated implementations of lattice reduction
algorithms using NVIDIA CUDA and cuBLAS.
"""

__version__ = "0.1.0"
__author__ = "cuBLASter Team"
__description__ = "CUDA-accelerated lattice reduction library"

# Try to import core functionality
try:
    from cublaster_core import (
        init_cuda,
        cleanup_cuda,
        get_memory_info,
        matmul,
        left_matmul,
        right_matmul,
        CudaLatticeReducer
    )
    
    # Initialize CUDA by default
    _cuda_available = True
    try:
        init_cuda()
    except Exception as e:
        print(f"Warning: CUDA initialization failed: {e}")
        _cuda_available = False
        
except ImportError as e:
    print(f"Warning: cuBLASter core not available: {e}")
    print("Please build the package with: pip install -e .")
    _cuda_available = False
    
    # Provide dummy implementations
    def init_cuda(*args, **kwargs):
        raise RuntimeError("cuBLASter core not compiled")
    
    def cleanup_cuda(*args, **kwargs):
        pass
    
    def get_memory_info(*args, **kwargs):
        raise RuntimeError("cuBLASter core not compiled")
    
    def matmul(*args, **kwargs):
        raise RuntimeError("cuBLASter core not compiled")
    
    def left_matmul(*args, **kwargs):
        raise RuntimeError("cuBLASter core not compiled")
    
    def right_matmul(*args, **kwargs):
        raise RuntimeError("cuBLASter core not compiled")
    
    class CudaLatticeReducer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cuBLASter core not compiled")

__all__ = [
    'init_cuda',
    'cleanup_cuda', 
    'get_memory_info',
    'matmul',
    'left_matmul',
    'right_matmul',
    'CudaLatticeReducer',
    '_cuda_available'
]
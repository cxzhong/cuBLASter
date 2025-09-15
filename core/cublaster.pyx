# distutils: language = c++
# Note: The actual source file (CUDA or CPU fallback) is determined at build time

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t

cimport cython
from cysignals.signals cimport sig_on, sig_off

from core.decl cimport *

# Initialize numpy
cnp.import_array()

# Global flag to track initialization
cdef bint _cuda_initialized = False

def init_cuda(int device_id=0):
    """Initialize CUDA and cuBLAS with specified device"""
    global _cuda_initialized
    if not _cuda_initialized:
        sig_on()
        cuda_init(device_id)
        sig_off()
        _cuda_initialized = True

def cleanup_cuda():
    """Cleanup CUDA resources"""
    global _cuda_initialized
    if _cuda_initialized:
        cuda_cleanup()
        _cuda_initialized = False

def get_memory_info():
    """Get GPU memory information"""
    cdef size_t free_mem, total_mem
    if not _cuda_initialized:
        init_cuda()
    
    sig_on()
    cuda_memory_info(&free_mem, &total_mem)
    sig_off()
    
    return {
        'free': free_mem,
        'total': total_mem,
        'used': total_mem - free_mem,
        'free_mb': free_mem // (1024 * 1024),
        'total_mb': total_mem // (1024 * 1024),
        'used_mb': (total_mem - free_mem) // (1024 * 1024)
    }

@cython.boundscheck(False)
@cython.wraparound(False)
def matmul(cnp.ndarray[ZZ, ndim=2, mode="c"] a, 
          cnp.ndarray[ZZ, ndim=2, mode="c"] b):
    """
    Compute matrix multiplication C = A * B using CUDA/cuBLAS
    
    Args:
        a: Input matrix A of shape (n, m)
        b: Input matrix B of shape (m, k)
        
    Returns:
        Matrix C of shape (n, k) where C = A * B
    """
    if not _cuda_initialized:
        init_cuda()
    
    cdef int n = a.shape[0]
    cdef int m = a.shape[1]
    cdef int k = b.shape[1]
    
    if b.shape[0] != m:
        raise ValueError(f"Matrix dimensions incompatible: ({n}, {m}) * ({b.shape[0]}, {k})")
    
    # Create output matrix
    cdef cnp.ndarray[ZZ, ndim=2, mode="c"] c = np.zeros((n, k), dtype=np.int64, order='C')
    
    sig_on()
    cuda_matmul(&a[0, 0], &b[0, 0], &c[0, 0], n, m, k)
    sig_off()
    
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
def left_matmul(cnp.ndarray[ZZ, ndim=2, mode="c"] a,
               cnp.ndarray[ZZ, ndim=2, mode="c"] b):
    """
    Compute in-place left matrix multiplication: B = A * B using CUDA/cuBLAS
    
    Args:
        a: Square matrix A of shape (n, n)
        b: Matrix B of shape (n, m) - modified in place
    """
    if not _cuda_initialized:
        init_cuda()
    
    cdef int n = a.shape[0]
    cdef int m = b.shape[1]
    
    if a.shape[1] != n or b.shape[0] != n:
        raise ValueError(f"Matrix dimensions incompatible for left multiplication")
    
    cdef int stride_a = a.strides[0] // sizeof(ZZ)
    cdef int stride_b = b.strides[0] // sizeof(ZZ)
    
    sig_on()
    cuda_left_matmul(&a[0, 0], &b[0, 0], n, m, stride_a, stride_b)
    sig_off()

@cython.boundscheck(False)
@cython.wraparound(False)
def right_matmul(cnp.ndarray[ZZ, ndim=2, mode="c"] a,
                cnp.ndarray[ZZ, ndim=2, mode="c"] b):
    """
    Compute in-place right matrix multiplication: A = A * B using CUDA/cuBLAS
    
    Args:
        a: Matrix A of shape (n, m) - modified in place
        b: Square matrix B of shape (m, m)
    """
    if not _cuda_initialized:
        init_cuda()
    
    cdef int n = a.shape[0]
    cdef int m = a.shape[1]
    
    if b.shape[0] != m or b.shape[1] != m:
        raise ValueError(f"Matrix dimensions incompatible for right multiplication")
    
    cdef int stride_a = a.strides[0] // sizeof(ZZ)
    
    sig_on()
    cuda_right_matmul_strided(&a[0, 0], &b[0, 0], n, m, stride_a)
    sig_off()

# Lattice reduction functionality (placeholder for now)
class CudaLatticeReducer:
    """CUDA-accelerated lattice reducer"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        init_cuda(device_id)
        
    def __del__(self):
        cleanup_cuda()
        
    def get_memory_info(self):
        """Get GPU memory information"""
        return get_memory_info()
        
    def reduce(self, lattice):
        """Perform lattice reduction (placeholder)"""
        # This will be implemented with the actual lattice reduction algorithms
        # For now, just return the input lattice
        return lattice
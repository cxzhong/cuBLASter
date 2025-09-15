#!/usr/bin/env python3
"""
Simple cuBLASter example demonstrating CUDA-accelerated matrix operations.
"""

import sys
import numpy as np

try:
    import cublaster
    from cublaster import matmul, get_memory_info, init_cuda
except ImportError:
    print("Error: cuBLASter not installed. Please run: pip install -e .")
    sys.exit(1)


def main():
    print("cuBLASter Simple Example")
    print("=" * 40)
    
    # Initialize CUDA
    try:
        init_cuda(device_id=0)
        print("✓ CUDA initialized successfully")
    except Exception as e:
        print(f"✗ CUDA initialization failed: {e}")
        return 1
    
    # Show GPU memory info
    try:
        mem_info = get_memory_info()
        print(f"✓ GPU Memory: {mem_info['used_mb']}MB used / {mem_info['total_mb']}MB total")
    except Exception as e:
        print(f"✗ Memory info failed: {e}")
        return 1
    
    # Test matrix multiplication
    print("\nTesting CUDA matrix multiplication...")
    
    # Create test matrices
    n, m, k = 4, 3, 5
    A = np.random.randint(-10, 10, size=(n, m), dtype=np.int64)
    B = np.random.randint(-10, 10, size=(m, k), dtype=np.int64)
    
    print(f"Matrix A ({n}x{m}):")
    print(A)
    print(f"Matrix B ({m}x{k}):")
    print(B)
    
    # Compute using cuBLAS
    try:
        C_cuda = matmul(A, B)
        print(f"Result C = A * B (CUDA):")
        print(C_cuda)
    except Exception as e:
        print(f"✗ CUDA matrix multiplication failed: {e}")
        return 1
    
    # Verify with NumPy
    C_numpy = A @ B
    print(f"Result C = A * B (NumPy):")
    print(C_numpy)
    
    # Check if results match
    if np.allclose(C_cuda.astype(float), C_numpy.astype(float)):
        print("✓ Results match! CUDA implementation is correct.")
    else:
        print("✗ Results don't match!")
        print(f"Difference: {np.abs(C_cuda - C_numpy).max()}")
        return 1
    
    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#ifndef CORE_CUDA_INTERFACE_HPP
#define CORE_CUDA_INTERFACE_HPP

#include "types.hpp"
#include <cstddef>

extern "C" {
    /**
     * Initialize CUDA and cuBLAS (or CPU fallback)
     */
    void cuda_init(int device_id = 0);

    /**
     * Cleanup CUDA resources
     */
    void cuda_cleanup();

    /**
     * Get GPU memory info (or dummy values for CPU fallback)
     */
    void cuda_memory_info(size_t* free_mem, size_t* total_mem);

    /**
     * Compute the matrix product between a and b, and store the result `a * b` in `c`.
     * Dimensions of `a`, `b` and `c` are assumed to be `n x m`, `m x k` and `n x k` respectively.
     */
    void cuda_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k);

    /**
     * Compute the matrix product between a and b, and store the result `a * b` in `b`.
     * Dimensions of `a` and `b` are assumed to be `n x n` and `n x m` respectively.
     */
    void cuda_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b);

    /**
     * Compute the matrix product between a and b, and store the result `a * b` in `a`.
     * Dimensions of `a` and `b` are assumed to be `n x m` and `m x m` respectively.
     */
    void cuda_right_matmul(ZZ *a, const ZZ *b, int n, int m);

    /**
     * Compute the matrix product with stride support
     */
    void cuda_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a);
}

#endif // CORE_CUDA_INTERFACE_HPP
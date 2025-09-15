#include <iostream>
#include <stdexcept>
#include <cstring>

#include "types.hpp"
#include "cuda_interface.hpp"

extern "C" {

/**
 * CPU fallback implementations when CUDA is not available
 */

void cuda_init(int device_id) {
    std::cout << "Warning: CUDA not available, using CPU fallback" << std::endl;
}

void cuda_cleanup() {
    // Nothing to cleanup for CPU fallback
}

void cuda_memory_info(size_t* free_mem, size_t* total_mem) {
    // Return dummy values for CPU fallback
    *free_mem = 1024 * 1024 * 1024;  // 1GB
    *total_mem = 1024 * 1024 * 1024; // 1GB
}

/**
 * Simple CPU matrix multiplication: C = A * B
 */
void cuda_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) {
    // Zero out result matrix
    memset(c, 0, n * k * sizeof(ZZ));
    
    // Perform matrix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < m; l++) {
                c[i * k + j] += a[i * m + l] * b[l * k + j];
            }
        }
    }
}

/**
 * CPU left matrix multiplication: B = A * B
 */
void cuda_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b) {
    ZZ *temp = new ZZ[n * m];
    memset(temp, 0, n * m * sizeof(ZZ));
    
    // Compute temp = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < n; l++) {
                temp[i * m + j] += a[i * stride_a + l] * b[l * stride_b + j];
            }
        }
    }
    
    // Copy result back to B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * stride_b + j] = temp[i * m + j];
        }
    }
    
    delete[] temp;
}

/**
 * CPU right matrix multiplication: A = A * B
 */
void cuda_right_matmul(ZZ *a, const ZZ *b, int n, int m) {
    ZZ *temp = new ZZ[n * m];
    memset(temp, 0, n * m * sizeof(ZZ));
    
    // Compute temp = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < m; l++) {
                temp[i * m + j] += a[i * m + l] * b[l * m + j];
            }
        }
    }
    
    // Copy result back to A
    memcpy(a, temp, n * m * sizeof(ZZ));
    delete[] temp;
}

void cuda_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a) {
    ZZ *temp = new ZZ[n * m];
    memset(temp, 0, n * m * sizeof(ZZ));
    
    // Compute temp = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < m; l++) {
                temp[i * m + j] += a[i * stride_a + l] * b[l * m + j];
            }
        }
    }
    
    // Copy result back to A with stride
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i * stride_a + j] = temp[i * m + j];
        }
    }
    
    delete[] temp;
}

} // extern "C"
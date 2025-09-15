#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>

#include "types.hpp"

// Global cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;
static bool cuda_initialized = false;

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        } \
    } while(0)

extern "C" {

/**
 * Initialize CUDA and cuBLAS
 */
void cuda_init(int device_id = 0) {
    if (cuda_initialized) {
        return;
    }
    
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Using GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    
    cuda_initialized = true;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup() {
    if (cuda_initialized && cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
        cuda_initialized = false;
    }
}

/**
 * Get GPU memory info
 */
void cuda_memory_info(size_t* free_mem, size_t* total_mem) {
    CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `c`.
 * All matrices are stored in row-major order on host.
 * Dimensions of `a`, `b` and `c` are assumed to be `n x m`, `m x k` and `n x k` respectively.
 */
void cuda_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) {
    if (!cuda_initialized) {
        cuda_init();
    }
    
    // Since cuBLAS operates on column-major matrices and we have row-major,
    // we compute C^T = B^T * A^T, then transpose the result
    // This gives us C = A * B as desired
    
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    size_t size_a = n * m * sizeof(double);
    size_t size_b = m * k * sizeof(double);
    size_t size_c = n * k * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));
    
    // Convert ZZ to double and copy to device
    double *h_a = new double[n * m];
    double *h_b = new double[m * k];
    
    for (int i = 0; i < n * m; i++) {
        h_a[i] = static_cast<double>(a[i]);
    }
    for (int i = 0; i < m * k; i++) {
        h_b[i] = static_cast<double>(b[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication: C = A * B
    // Using cuBLAS DGEMM: C = alpha * op(A) * op(B) + beta * C
    const double alpha = 1.0, beta = 0.0;
    
    // Since cuBLAS expects column-major and we have row-major,
    // we compute C^T = B^T * A^T by calling gemm with B^T and A^T
    CUBLAS_CHECK(cublasDgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            k, n, m,
                            &alpha,
                            d_b, k,  // B^T is k x m
                            d_a, m,  // A^T is m x n
                            &beta,
                            d_c, k));  // C^T is k x n
    
    // Copy result back to host as double
    double *h_c = new double[n * k];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));
    
    // Convert back to ZZ and transpose (since we computed C^T)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = static_cast<ZZ>(h_c[j * n + i]);
        }
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `b`.
 * Dimensions of `a` and `b` are assumed to be `n x n` and `n x m` respectively.
 */
void cuda_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b) {
    if (!cuda_initialized) {
        cuda_init();
    }
    
    // Allocate temporary memory for result
    ZZ *temp = new ZZ[n * m];
    
    // Extract matrices with strides
    double *h_a = new double[n * n];
    double *h_b = new double[n * m];
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[i * n + j] = static_cast<double>(a[i * stride_a + j]);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            h_b[i * m + j] = static_cast<double>(b[i * stride_b + j]);
        }
    }
    
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    size_t size_a = n * n * sizeof(double);
    size_t size_b = n * m * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_b));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication
    const double alpha = 1.0, beta = 0.0;
    
    CUBLAS_CHECK(cublasDgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, n,
                            &alpha,
                            d_b, m,
                            d_a, n,
                            &beta,
                            d_c, m));
    
    // Copy result back
    double *h_c = new double[n * m];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size_b, cudaMemcpyDeviceToHost));
    
    // Convert back and store with stride
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * stride_b + j] = static_cast<ZZ>(h_c[j * n + i]);
        }
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `a`.
 * Dimensions of `a` and `b` are assumed to be `n x m` and `m x m` respectively.
 */
void cuda_right_matmul(ZZ *a, const ZZ *b, int n, int m) {
    if (!cuda_initialized) {
        cuda_init();
    }
    
    // Extract matrices
    double *h_a = new double[n * m];
    double *h_b = new double[m * m];
    
    for (int i = 0; i < n * m; i++) {
        h_a[i] = static_cast<double>(a[i]);
    }
    for (int i = 0; i < m * m; i++) {
        h_b[i] = static_cast<double>(b[i]);
    }
    
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    size_t size_a = n * m * sizeof(double);
    size_t size_b = m * m * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_a));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication: C = A * B
    const double alpha = 1.0, beta = 0.0;
    
    CUBLAS_CHECK(cublasDgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, m,
                            &alpha,
                            d_b, m,
                            d_a, m,
                            &beta,
                            d_c, m));
    
    // Copy result back
    double *h_c = new double[n * m];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size_a, cudaMemcpyDeviceToHost));
    
    // Convert back and store
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i * m + j] = static_cast<ZZ>(h_c[j * n + i]);
        }
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void cuda_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a) {
    if (!cuda_initialized) {
        cuda_init();
    }
    
    // Extract matrices with strides
    double *h_a = new double[n * m];
    double *h_b = new double[m * m];
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            h_a[i * m + j] = static_cast<double>(a[i * stride_a + j]);
        }
    }
    for (int i = 0; i < m * m; i++) {
        h_b[i] = static_cast<double>(b[i]);
    }
    
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    size_t size_a = n * m * sizeof(double);
    size_t size_b = m * m * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_a));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication: C = A * B
    const double alpha = 1.0, beta = 0.0;
    
    CUBLAS_CHECK(cublasDgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, m,
                            &alpha,
                            d_b, m,
                            d_a, m,
                            &beta,
                            d_c, m));
    
    // Copy result back
    double *h_c = new double[n * m];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size_a, cudaMemcpyDeviceToHost));
    
    // Convert back and store with stride
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i * stride_a + j] = static_cast<ZZ>(h_c[j * n + i]);
        }
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

} // extern "C"
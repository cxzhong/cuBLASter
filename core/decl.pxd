# Cython declarations for cuBLASter core functionality

cdef extern from "types.hpp":
    ctypedef double FT
    ctypedef long long ZZ
    cdef int MAX_ENUM_N

# These functions are available in both CUDA and CPU fallback versions
cdef extern from *:
    void cuda_init(int device_id) except +
    void cuda_cleanup() except +
    void cuda_memory_info(size_t* free_mem, size_t* total_mem) except +
    void cuda_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) except +
    void cuda_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b) except +
    void cuda_right_matmul(ZZ *a, const ZZ *b, int n, int m) except +
    void cuda_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a) except +
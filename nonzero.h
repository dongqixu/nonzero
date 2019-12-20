#ifndef NONZERO_NONZERO_H
#define NONZERO_NONZERO_H

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef CUDA_1D_KERNEL_LOOP
#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)
#endif //CUDA_1D_KERNEL_LOOP

template<typename T>
int devicePrint(const T *deviceValues, int length, const std::string &info, int step);

template<typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
    return (a + b - 1) / b;
}

cudaError_t nonzero(float *scores, float *boxes, int length);

#endif //NONZERO_NONZERO_H

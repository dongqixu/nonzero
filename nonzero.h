#ifndef NONZERO_NONZERO_H
#define NONZERO_NONZERO_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

cudaError_t nonzero(float *scores, float *boxes, int length);

#endif //NONZERO_NONZERO_H

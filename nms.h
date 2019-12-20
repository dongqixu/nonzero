#ifndef NONZERO_NMS_H
#define NONZERO_NMS_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef CUDA_1D_KERNEL_LOOP
#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)
#endif //CUDA_1D_KERNEL_LOOP

cudaError_t nms(float *boxes, float *scores, int length, float threshold);

#endif //NONZERO_NMS_H

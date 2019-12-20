#ifndef CUB_STDERR
#define CUB_STDERR
#endif //CUB_STDERR
#ifndef VERBOSE
#define VERBOSE
#endif //VERBOSE

#include <cub/cub.cuh>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "nms.h"
#include "nonzero.h"

const int nmsThreadsPerBlock = sizeof(unsigned long long) * 8;

template<typename T>
__device__ inline T calculateIoU(T const *const a, T const *const b) {
    T left = max(a[0], b[0]), right = min(a[2], b[2]);
    T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
    T interS = width * height;
    T Sa = (a[2] - a[0]) * (a[3] - a[1]);
    T Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / (Sa + Sb - interS);
}

template<typename T>
__global__ void initKernel(T *outputs, int64_t length) {
    CUDA_1D_KERNEL_LOOP(index, length) {
        outputs[index] = (T)index;
    }
}

template<typename T>
__global__ void gatherKernel(const T *inputs, const int *indices, T *outputs, int64_t length, int step) {
    CUDA_1D_KERNEL_LOOP(index, length) {
        for (int i = 0; i < step; i++) {
            outputs[index * step + i] = inputs[indices[index] * step + i];
        }
    }
}

template<typename T>
__global__ void nmsKernel(const T *boxes, const int boxNum, const float threshold, unsigned long long *mask) {
    const int rowIndex = blockIdx.y;
    const int colIndex = blockIdx.x;
    const int rowSize = min(boxNum - rowIndex * nmsThreadsPerBlock, nmsThreadsPerBlock);
    const int colSize = min(boxNum - colIndex * nmsThreadsPerBlock, nmsThreadsPerBlock);

    __shared__ T blockBoxes[nmsThreadsPerBlock * 4];
    if (threadIdx.x < colSize) {
        for (int i = 0; i < 4; i++) {
            blockBoxes[threadIdx.x * 4 + i] = boxes[(colIndex * nmsThreadsPerBlock + threadIdx.x) * 4 + i];
        }
    }
    __syncthreads();

    if (threadIdx.x < rowSize) {
        const int currentBoxIndex = rowIndex * nmsThreadsPerBlock + threadIdx.x;
        const T *currentBox = boxes + currentBoxIndex * 4;
        // use data type "unsigned long long" to represent the mask
        unsigned long long t = 0;
        int start = (rowIndex == colIndex) ? threadIdx.x + 1 : 0;
        for (int i = start; i < colSize; i++) {
            if (calculateIoU(currentBox, blockBoxes + i * 4) > threshold) {
                t |= 1ULL << i;
            }
        }
        // mask shape: (boxNum, colBlocks)
        const int colBlocks = ATenCeilDiv(boxNum, nmsThreadsPerBlock);
        mask[currentBoxIndex * colBlocks + colIndex] = t;
    }
}

cudaError_t nms(float *boxes, float *scores, int length, float threshold) {
    // copy data to device
    float *deviceBoxes, *deviceScores;
    cudaMalloc(&deviceBoxes, sizeof(float) * length * 4);
    cudaMalloc(&deviceScores, sizeof(float) * length);
    cudaMemcpy(deviceBoxes, boxes, sizeof(float) * length * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceScores, scores, sizeof(float) * length, cudaMemcpyHostToDevice);

    // nms logic begins
    // inputs: *deviceBoxes, *deviceScores, length, threshold
    // 1. Detections sort by scores 
    // prepare score sorting output memory
    float *deviceSortedScores;
    int *deviceCounting, *deviceSortedIndex;
    cudaMalloc(&deviceSortedScores, sizeof(float) * length);
    cudaMalloc(&deviceCounting, sizeof(int) * length);
    cudaMalloc(&deviceSortedIndex, sizeof(int) * length);

    // initialize indices
    int blockNum = ATenCeilDiv(length, nmsThreadsPerBlock);
    dim3 grid(blockNum);
    dim3 block(nmsThreadsPerBlock);
    initKernel<<<grid, block>>>(deviceCounting, length);

    // sort socores and return indices
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, deviceScores, deviceSortedScores,
            deviceCounting, deviceSortedIndex, length);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, deviceScores, deviceSortedScores,
            deviceCounting, deviceSortedIndex, length);
    cudaFree(d_temp_storage);
    cudaFree(deviceCounting);

    // index select sorted_detections
    float *deviceSortedBoxes;
    cudaMalloc(&deviceSortedBoxes, sizeof(float) * length * 4);
    gatherKernel<<<grid, block>>>(deviceBoxes, deviceSortedIndex, deviceSortedBoxes, length, 4);

#ifdef VERBOSE
    devicePrint(deviceSortedIndex, length, std::string("deviceSortedIndex"), 1);
#endif

#ifdef DEBUG
    devicePrint(deviceSortedScores, length, std::string("deviceSortedScores"), 1);
    devicePrint(deviceSortedBoxes, length * 4, std::string("deviceSortedBoxes"), 4);
#endif

    // 2. create IoU lookup table
    unsigned long long *deviceMask;
    cudaMalloc(&deviceMask, sizeof(unsigned long long) * length * blockNum);
    dim3 nmsBlocks(blockNum, blockNum);
    dim3 nmsThreads(nmsThreadsPerBlock);
    nmsKernel<<<nmsBlocks, nmsThreads>>>(deviceSortedBoxes, length, threshold, deviceMask);

    unsigned long long *mask = (unsigned long long *)malloc(sizeof(unsigned long long) * length * blockNum);
    cudaMemcpy(mask, deviceMask, sizeof(unsigned long long) * length * blockNum, cudaMemcpyDeviceToHost);
    cudaFree(deviceMask);

#ifdef DEBUG
    std::cout << "mask: " << std::endl;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < blockNum; j++) {
            for (unsigned k = 0; k < std::min(length - j * nmsThreadsPerBlock, nmsThreadsPerBlock); k++) {
                unsigned long long bit = (mask[i * blockNum + j] >> k & 1);
                std::cout << bit << " ";
            }
        }
        std::cout << std::endl;
    }
#endif

    // 3. apply suppression on cpu
    unsigned long long *remove = (unsigned long long *)malloc(sizeof(unsigned long long) * blockNum);
    memset(remove, 0, sizeof(unsigned long long) * blockNum);

    int *keep = (int *)malloc(sizeof(int) * length);
    memset(keep, 0, sizeof(int) * length);

    int numKeep = 0;
    for (int i = 0; i < length; i++) {
        int nBlock = i / nmsThreadsPerBlock;
        int inBlock = i % nmsThreadsPerBlock;

        if (!(remove[nBlock] & (1ULL << (unsigned)inBlock))) {
            keep[numKeep++] = i;
            unsigned long long *p = mask + i * blockNum;
            for (int j = nBlock; j < blockNum; j++) {
                remove[j] |= p[j];
            }
        }
    }

#ifdef DEBUG
    std::cout << "keep" << ": ";
    for (int i = 0; i < numKeep; i++) {
        std::cout << keep[i] << " ";
    }
    std::cout << std::endl;
#endif

    int *deviceKeep, *deviceOutput;
    cudaMalloc(&deviceKeep, sizeof(int) * numKeep);
    cudaMalloc(&deviceOutput, sizeof(int) * numKeep);
    cudaMemcpy(deviceKeep, keep, sizeof(int) * numKeep, cudaMemcpyHostToDevice);

#ifdef VERBOSE
    devicePrint(deviceKeep, numKeep, std::string("deviceKeep"), 1);
#endif

    gatherKernel<<<grid, block>>>(deviceSortedIndex, deviceKeep, deviceOutput, numKeep, 1);

#ifdef VERBOSE
    devicePrint(deviceOutput, numKeep, std::string("deviceOutput"), 1);
    std::cout << "numKeep: " << numKeep << std::endl;
#endif

    cudaFree(deviceSortedScores);
    cudaFree(deviceSortedIndex);
    cudaFree(deviceSortedBoxes);
    cudaFree(deviceScores);
    cudaFree(deviceBoxes);
    cudaFree(deviceKeep);
    cudaFree(deviceOutput);
    free(mask);
    free(keep);
    return cudaGetLastError();
}

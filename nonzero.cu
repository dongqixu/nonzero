#ifndef CUB_STDERR
#define CUB_STDERR
#endif //CUB_STDERR
#ifndef VERBOSE
#define VERBOSE
#endif //VERBOSE

#include <cub/cub.cuh>
#include <stdio.h>
#include <iostream>
#include "nonzero.h"

const int RUNTIME_BLOCK_THREADS = 8;

// sum number of nonzero elements in each block with reduce
template<typename T, int BLOCK_THREADS>
__global__ void nonZeroCountKernel(const T *inputs, int64_t length, int *counts) {
    typedef cub::BlockReduce<int, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    cub::CastOp<bool> cast_to_bool;
    int num = 0;
    if (index < length && cast_to_bool(inputs[index])) {
        num++;
    }
    int aggregate = BlockReduceT(temp_storage).Sum(num);
    // record aggregated count number by thread 0
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = aggregate;
    }
}

// write nonzero element indices to output
template<typename T, int BLOCK_THREADS>
__global__ void nonZeroIndexKernel(const T *inputs, int64_t length, const int *cumulativeCounts, int *output) {
    typedef cub::BlockScan<int, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    cub::CastOp<bool> cast_to_bool;
    int num = 0, nonzero = 0;
    if (index < length && cast_to_bool(inputs[index])) {
        num++;
    }
    // exclusive sum number of nonzero elements in block as output index
    BlockScanT(temp_storage).ExclusiveSum(num, nonzero);
    if (index < length && cast_to_bool(inputs[index])) {
        int offset = blockIdx.x ? cumulativeCounts[blockIdx.x - 1] : 0;
        output[offset + nonzero] = index;
    }
}

// gather inputs with given indices and element dimension
template<typename T>
__global__ void gatherKernel(const T *inputs, const int *indices, T *outputs, int64_t length, int step) {
    CUDA_1D_KERNEL_LOOP(index, length) {
        for (int i = 0; i < step; i++) {
            outputs[index * step + i] = inputs[indices[index] * step + i];
        }
    }
}

// cout device elements 
template<typename T>
int devicePrint(const T *deviceValues, int length, const std::string &info, int step) {
    T *values = (T *)malloc(sizeof(T) * length);
    cudaMemcpy(values, deviceValues, sizeof(T) * length, cudaMemcpyDeviceToHost);
    std::cout << info << ": ";
    for (int i = 0; i < length; i++) {
        if (step != 1) {
            if (!(i % step)) {
                std::cout << std::endl;
            }
        }
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
    free(values);
    return 0;
}

// Apply nonzero opeartion to scores, gather scores and boxes with returned nonzero indices.
cudaError_t nonzero(float *scores, float *boxes, int length) {
    // inputs: copy scores and boxes to device
    float *deviceScores, *deviceBoxes;
    cudaMalloc(&deviceScores, sizeof(float) * length);
    cudaMalloc(&deviceBoxes, sizeof(float) * length * 4);
    cudaMemcpy(deviceScores, scores, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBoxes, boxes, sizeof(float) * length * 4, cudaMemcpyHostToDevice);

    // malloc space for nonzero block counts & cumulative counts
    int blockNum = ATenCeilDiv(length, RUNTIME_BLOCK_THREADS);
    int *devicesCounts, *devicesCumulativeCounts;
    cudaMalloc(&devicesCounts, sizeof(int) * blockNum);
    cudaMalloc(&devicesCumulativeCounts, sizeof(int) * blockNum);

    // count number of nonzero elements by block
    dim3 grid(blockNum);
    dim3 block(RUNTIME_BLOCK_THREADS);
    nonZeroCountKernel<float, RUNTIME_BLOCK_THREADS><<<grid, block>>>(deviceScores, (int64_t)length, devicesCounts);
#ifdef VERBOSE
    devicePrint(devicesCounts, blockNum, std::string("devicesCounts"), 1);
#endif

    // inclusive sum number of nonzero elements for each block
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devicesCounts, devicesCumulativeCounts, blockNum);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devicesCounts, devicesCumulativeCounts, blockNum);
    // cudaFree(d_temp_storage);

#ifdef VERBOSE
    devicePrint(devicesCumulativeCounts, blockNum, std::string("devicesCumulativeCounts"), 1);
#endif

    // retrieve nonzero indices
    int nonzeroCount, *nonzeroIndex;
    cudaMemcpy(&nonzeroCount, devicesCumulativeCounts + blockNum - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&nonzeroIndex, sizeof(int) * nonzeroCount);
    nonZeroIndexKernel<float, RUNTIME_BLOCK_THREADS><<<grid, block>>>(
        deviceScores, (int64_t)length, devicesCumulativeCounts, nonzeroIndex);

#ifdef VERBOSE
    std::cout << "nonzeroCount: " << nonzeroCount << std::endl;
    devicePrint(nonzeroIndex, nonzeroCount, std::string("nonzeroIndex"), 1);
#endif

    // retrieve nonzero elements
    float *outputScores, *outputBoxes;
    cudaMalloc(&outputScores, sizeof(float) * nonzeroCount);
    cudaMalloc(&outputBoxes, sizeof(float) * nonzeroCount * 4);
    gatherKernel<<<grid, block>>>(deviceScores, nonzeroIndex, outputScores, nonzeroCount, 1);
    gatherKernel<<<grid, block>>>(deviceBoxes, nonzeroIndex, outputBoxes, nonzeroCount, 4);

#ifdef VERBOSE
    std::cout << "Outputs: " << std::endl;
    devicePrint(outputScores, nonzeroCount, std::string("scores"), 1);
    devicePrint(outputBoxes, nonzeroCount * 4, std::string("boxes"), 4);
#endif

    // release memory
    cudaFree(deviceScores);
    cudaFree(deviceBoxes);
    cudaFree(devicesCounts);
    cudaFree(devicesCumulativeCounts);
    cudaFree(d_temp_storage);
    cudaFree(nonzeroIndex);
    cudaFree(outputScores);
    cudaFree(outputBoxes);
    return cudaGetLastError();
}

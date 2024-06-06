
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>

#define X_DIM 1024
#define Y_DIM 1024

__global__ void compute_histogram_renamed(int* img_data, int* hist, int length) {
    // Calculate the global thread index
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory to avoid bank conflicts and reduce atomic operations
    __shared__ int shared_hist[256];

    // Initialize shared histogram
    if (threadIdx.x < 256) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread updates the shared histogram
    if (global_tid < length) {
        atomicAdd(&shared_hist[img_data[global_tid]], 1);
    }
    __syncthreads();

    // Transfer the shared histogram to the global histogram
    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], shared_hist[threadIdx.x]);
    }
}

int main() {
    int length = X_DIM * Y_DIM;
    int *host_img_data = (int*)malloc(length * sizeof(int));
    int *host_hist = (int*)calloc(256, sizeof(int));

    std::string line = "";
    std::ifstream inputFile("unhappy.txt");

    for (int i = 0; i < length; ++i) {
        if (!(inputFile >> line)) {
            std::cerr << "Error reading data at index " << i << std::endl;
            return -1;
        }
        try {
            host_img_data[i] = std::stoi(line);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid input at index " << i << ": " << line << std::endl;
            return -1;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range input at index " << i << ": " << line << std::endl;
            return -1;
        }
    }

    int *device_img_data;
    int *device_hist;

    cudaMalloc((void**)&device_img_data, length * sizeof(int));
    cudaMalloc((void**)&device_hist, 256 * sizeof(int));

    cudaMemcpy(device_img_data, host_img_data, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(device_hist, 0, 256 * sizeof(int));

    int threads_per_block = 256;
    int num_blocks = (length + threads_per_block - 1) / threads_per_block;

    compute_histogram_renamed<<<num_blocks, threads_per_block>>>(device_img_data, device_hist, length);

    cudaMemcpy(host_hist, device_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream outputHistogramFile("histogram.txt");
    for (int i = 0; i < 256; i++)
        outputHistogramFile << host_hist[i] << "\n";

    free(host_img_data);
    free(host_hist);
    cudaFree(device_img_data);
    cudaFree(device_hist);

    return 0;
}



#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <math.h>

const int N = 512;
const int M = 512;
const int P_MEDIAN = 7;
const int P_GAUSSIAN = 11;

__device__ int getGid() {
    int tid = threadIdx.x + blockDim.x * threadIdx.y +
              blockDim.x * blockDim.y * threadIdx.z;
    int bid = blockIdx.x + gridDim.x * blockIdx.y +
              gridDim.x * gridDim.y * blockIdx.z;
    int gid = bid * blockDim.x * blockDim.y * blockDim.z + tid;
    return gid;
}

__device__ void Array_sort(float *array, int n) {
    int i = 0, j = 0;
    float temp = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n - 1; j++) {
            if (array[j] > array[j + 1]) {
                temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

__device__ float Array_sum(float *array, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += array[i];
    return sum;
}

__global__ void medianFilter(float* im, int size, int channel_offset) {
    size_t gid = getGid();
    float value = 0.0;
    int channel_size = size / 3;

    if (gid < channel_size) {
        int offset = gid + channel_offset;

        if (((gid % M) >= P_MEDIAN) && ((gid % M) <= (M - P_MEDIAN - 1))) {
            float stencil[2 * P_MEDIAN + 1];
            for (int k = 0; k < 2 * P_MEDIAN + 1; k++)
                stencil[k] = im[offset - P_MEDIAN * 3 + k * 3];
            Array_sort(stencil, 2 * P_MEDIAN + 1);
            value = stencil[P_MEDIAN];
        }
        __syncthreads();
        im[offset] = value;
        __syncthreads();
    }

    if (gid < channel_size) {
        int offset = gid + channel_offset;

        if (((gid / M) >= P_MEDIAN) && ((gid / M) <= (N - P_MEDIAN - 1))) {
            float stencil[2 * P_MEDIAN + 1];
            for (int k = 0; k < 2 * P_MEDIAN + 1; k++)
                stencil[k] = im[offset + (k - P_MEDIAN) * M * 3];
            Array_sort(stencil, 2 * P_MEDIAN + 1);
            value = stencil[P_MEDIAN];
        }
        __syncthreads();
        im[offset] = value;
        __syncthreads();
    }
}

__global__ void gaussianBlur(float* im, int size, int channel_offset) {
    size_t gid = getGid();
    float sigma = 0.9;
    float G[2 * P_GAUSSIAN + 1];
    for (int k = 0; k < 2 * P_GAUSSIAN + 1; k++) {
        G[k] = 1 / sqrt(2 * 3.1415 * sigma) * exp(-(k - P_GAUSSIAN) * (k - P_GAUSSIAN) / (2 * sigma * sigma));
    }
    int channel_size = size / 3;

    if (gid < channel_size) {
        int offset = gid + channel_offset;
        float value = im[offset];

        if (((gid % M) >= P_GAUSSIAN) && ((gid % M) <= (M - P_GAUSSIAN - 1))) {
            float stencil[2 * P_GAUSSIAN + 1];
            for (int k = 0; k < 2 * P_GAUSSIAN + 1; k++)
                stencil[k] = im[offset - P_GAUSSIAN * 3 + k * 3] * G[k];
            value = Array_sum(stencil, 2 * P_GAUSSIAN + 1);
        }
        __syncthreads();
        im[offset] = value;
        __syncthreads();
    }

    if (gid < channel_size) {
        int offset = gid + channel_offset;
        float value = im[offset];

        if (((gid / M) >= P_GAUSSIAN) && ((gid / M) <= (N - P_GAUSSIAN - 1))) {
            float stencil[2 * P_GAUSSIAN + 1];
            for (int k = 0; k < 2 * P_GAUSSIAN + 1; k++)
                stencil[k] = im[offset + (k - P_GAUSSIAN) * M * 3] * G[k];
            value = Array_sum(stencil, 2 * P_GAUSSIAN + 1);
        }
        __syncthreads();
        im[offset] = value;
        __syncthreads();
    }
}

void processFilter(const std::string& inputFile, const std::string& medianOutputFile, const std::string& gaussianOutputFile) {
    std::string a = "";
    std::ifstream fileInput(inputFile);
    std::vector<float> image(N * M * 3);
    for (auto n = 0; n < N * M * 3; ++n) {
        fileInput >> a;
        image[n] = std::stof(a);
    }

    float* imageCuda;
    cudaMalloc(&imageCuda, N * M * 3 * sizeof(float));

    cudaMemcpy(imageCuda, image.data(), N * M * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform median filter for each channel
    for (int channel = 0; channel < 3; ++channel) {
        int channel_offset = channel * N * M;
        medianFilter<<<(N * M + 255) / 256, 256>>>(imageCuda, N * M * 3, channel_offset);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(image.data(), imageCuda, N * M * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream medianOutput(medianOutputFile);
    for (int i = 0; i < N * M * 3; i++)
        medianOutput << image[i] << "\n";

    // Reload image data for Gaussian blur
    fileInput.clear();
    fileInput.seekg(0, std::ios::beg);
    for (auto n = 0; n < N * M * 3; ++n) {
        fileInput >> a;
        image[n] = std::stof(a);
    }
    cudaMemcpy(imageCuda, image.data(), N * M * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform Gaussian blur for each channel
    for (int channel = 0; channel < 3; ++channel) {
        int channel_offset = channel * N * M;
        gaussianBlur<<<(N * M + 255) / 256, 256>>>(imageCuda, N * M * 3, channel_offset);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(image.data(), imageCuda, N * M * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream gaussianOutput(gaussianOutputFile);
    for (int i = 0; i < N * M * 3; i++)
        gaussianOutput << image[i] << "\n";

    cudaFree(imageCuda);
}

int main() {
    processFilter("unhappy_filter.txt", "median_filter.txt", "gaussian_blur.txt");
    return 0;
}


#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
using namespace std;


__global__ void compute_delta(double* data, double* delta, int x_steps, int y_steps, double hx, double hy) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = tid; idx < (x_steps-2)*(y_steps-2); idx += num_threads) {
        int xid = idx % (x_steps-2) + 1;
        int yid = idx / (x_steps-2) + 1;
        delta[xid + yid * x_steps] = (
            data[xid + 1 + yid * x_steps] - 2 * data[xid + yid * x_steps] + data[xid - 1 + yid * x_steps]
        ) / hx / hx + (
            data[xid + yid * x_steps + x_steps] - 2 * data[xid + yid * x_steps] + data[xid + yid * x_steps - x_steps]
        ) / hy / hy;
    }
}

__global__ void clear_memory(double* to_clear) {
    *to_clear = 0;
}

__global__ void add_with_tau(double* data, double* delta, double tau, int x_steps, int y_steps) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = tid; idx < (x_steps-2)*(y_steps-2); idx += num_threads) {
        int xid = idx % (x_steps-2) + 1;
        int yid = idx / (x_steps-2) + 1;
        data[xid + yid * x_steps] += tau * delta[xid + yid * x_steps];
    }
}

__global__ void compute_l1_norm(double* data1, double* data2, double* norm, int x_steps, int y_steps) {
    int num_threads = blockDim.x * gridDim.x;
    double candidate;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = tid; idx < (x_steps-2)*(y_steps-2); idx += num_threads) {
        int xid = idx % (x_steps-2) + 1;
        int yid = idx / (x_steps-2) + 1;
        candidate = abs(data1[xid + yid * x_steps] - data2[xid + yid * x_steps]); // Don't compare boundary since it's fixed
        if (candidate > *norm) {
            *norm = candidate;
        }
    }
}

__global__ void assign_values(double* target, double* source, int x_steps, int y_steps) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = tid; idx < (x_steps-2)*(y_steps-2); idx += num_threads) {
        int xid = idx % (x_steps-2) + 1;
        int yid = idx / (x_steps-2) + 1;
        target[xid + yid * x_steps] = source[xid + yid * x_steps];
    }
}

__global__ void set_boundary_conditions(double* data, int x_steps, int y_steps) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = tid; idx < (x_steps) * (y_steps); idx += num_threads) {
        int xid = idx % (x_steps);
        int yid = idx / (x_steps);
        data[xid + yid * x_steps] = (xid == 0) ? 0 : (xid == (x_steps - 1)) ? 0 : (yid == 0) ? 1 : (yid == (y_steps - 1)) ? 0 : 0;
    }
}

int main(int argc, char* argv[]) {
    std::ofstream output_file;
    output_file.open(argv[4]);
    size_t x_steps = atoi(argv[1]);
    size_t y_steps = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    double error = 0;
    double* error_control;
    double tau = atof(argv[5]);
    double *u_current, *u_next, *delta;
    int iteration_count = 0;
    double tolerance = 1e-16;
    cudaMallocManaged((void**)&u_current, x_steps * y_steps * sizeof(double));
    cudaMallocManaged((void**)&u_next, x_steps * y_steps * sizeof(double));
    cudaMallocManaged((void**)&delta, x_steps * y_steps * sizeof(double));
    cudaMallocManaged((void**)&error_control, sizeof(double));
    double hy = 1.0 / y_steps;
    double hx = 1.0 / x_steps;
    int num_blocks = 1 + (x_steps-2) * (y_steps-2) / num_threads;

    set_boundary_conditions <<<num_blocks, num_threads>>> (u_current, x_steps, y_steps);
    set_boundary_conditions <<<num_blocks, num_threads>>> (u_next, x_steps, y_steps);
    cudaDeviceSynchronize();

    error = 100;


    while (error > tolerance) {
        compute_delta <<<num_blocks, num_threads>>> (u_current, delta, x_steps, y_steps, hx, hy);
        assign_values <<<num_blocks, num_threads>>> (u_current, u_next, x_steps, y_steps);
        add_with_tau <<<num_blocks, num_threads>>> (u_next, delta, tau, x_steps, y_steps);
        cudaDeviceSynchronize();
        clear_memory <<<num_blocks, num_threads>>> (error_control);
        compute_l1_norm <<<num_blocks, num_threads>>> (u_next, u_current, error_control, x_steps, y_steps);
        cudaMemcpy(&error, error_control, sizeof(double), cudaMemcpyDeviceToHost);
        iteration_count++;
        if (iteration_count % 100 == 0) {
            std::cout << "Iteration number " << iteration_count << ", L1 norm " << error << std::endl;
        }
    }

    std::cout << error << std::endl;
    cudaDeviceSynchronize();

    for (int i = 0; i < x_steps; i++) {
        for (int j = 0; j < y_steps; j++) {
            output_file << u_next[i + j * x_steps] << ',';
        }
        output_file << std::endl;
    }


    cudaFree(u_current);
    cudaFree(u_next);
    cudaFree(delta);
    cudaFree(error_control);
    output_file.close();
}

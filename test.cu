#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define tile 16

__global__ void matrix_multiply(const float* A, const float* B, float* C, int n, int v, int k){
    __shared__ float A_tile[tile][tile];
    __shared__ float B_tile[tile][tile];

    int row = blockIdx.y*tile + threadIdx.y;
    int col = blockIdx.x*tile + threadIdx.x;
    float out = 0.f;

    for(int tile_idx=0;tile_idx<(v+tile-1)/tile; tile_idx++){
        int a_global_row = row;
        int a_global_col = tile_idx * tile + threadIdx.x;
        A_tile[threadIdx.y][threadIdx.x] = (a_global_row < n && a_global_col < v)? 
            A[a_global_row * v + a_global_col]: 0.0f;

        int b_global_row = tile_idx * tile + threadIdx.y;
        int b_global_col = col;
        B_tile[threadIdx.y][threadIdx.x] = (b_global_row < v && b_global_col < k)? 
            B[b_global_row * k + b_global_col]: 0.0f;

        __syncthreads();
        for (int i = 0; i < tile; i++) {
            out += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < k) {
        C[row * k + col] = out;
    }
}

void cpu_multiply(const float* A, const float* B, float* C, int n, int v, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < v; l++) {
                sum += A[i * v + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

bool test_case(int n, int v, int k) {
    printf("Testing %dx%d × %dx%d... ", n, v, v, k);
    
    // Allocate memory
    float *h_A = new float[n*v];
    float *h_B = new float[v*k]; 
    float *h_C_cpu = new float[n*k];
    float *h_C_gpu = new float[n*k];
    
    // Random data
    srand(42);
    for(int i = 0; i < n*v; i++) h_A[i] = rand() % 10;
    for(int i = 0; i < v*k; i++) h_B[i] = rand() % 10;
    
    // CPU
    clock_t start = clock();
    cpu_multiply(h_A, h_B, h_C_cpu, n, v, k);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    
    // GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n*v*sizeof(float));
    cudaMalloc(&d_B, v*k*sizeof(float));
    cudaMalloc(&d_C, n*k*sizeof(float));
    
    cudaMemcpy(d_A, h_A, n*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, v*k*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(tile, tile);
    dim3 grid((k+tile-1)/tile, (n+tile-1)/tile);
    
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    
    cudaEventRecord(gpu_start);
    matrix_multiply<<<grid, block>>>(d_A, d_B, d_C, n, v, k);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    cudaMemcpy(h_C_gpu, d_C, n*k*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check correctness
    bool correct = true;
    for(int i = 0; i < n*k; i++) {
        if(fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
            correct = false;
            break;
        }
    }
    
    long long ops = (long long)n * v * k * 2;
    double cpu_gflops = ops / (cpu_time/1000.0) / 1e9;
    double gpu_gflops = ops / (gpu_time/1000.0) / 1e9;
    
    printf("%s | CPU: %.1fms (%.1f GFLOPS) | GPU: %.1fms (%.1f GFLOPS) | Speedup: %.1fx\n",
           correct ? "✅" : "❌", cpu_time, cpu_gflops, gpu_time, gpu_gflops, cpu_time/gpu_time);
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(gpu_start); cudaEventDestroy(gpu_stop);
    
    return correct;
}

int main() {
    printf("Matrix Multiplication Test\n");
    printf("==========================\n");
    
    int tests[][3] = {
        {64, 64, 64},
        {128, 128, 128}, 
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024}
    };
    
    for(int i = 0; i < 5; i++) {
        test_case(tests[i][0], tests[i][1], tests[i][2]);
    }
    
    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>
#include "layers.cuh"
#include<vector>
#include<algorithm>

#define tile  16
//general batch wise matrix multiplication code;
/*
A consists of n*v; B consists of v*k; 
Implementing tiled matrix multiplication as that is most effecient
*/
__global__ void matrix_multiply(const float* A, const float* B, float* C, 
    int n, int v, int k){
    __shared__ float A_tile[tile][tile];
    __shared__ float B_tile[tile][tile];

    int row = blockIdx.y*tile + threadIdx.y;
    int col = blockIdx.x*tile + threadIdx.x;
    float out = 0.f;

    for(int tile_idx=0;tile_idx<(v+tile-1)/tile; tile_idx++){
        int a_tile_row = threadIdx.y;
        int a_tile_col = threadIdx.x;
        int a_global_row = row;
        int a_global_col = tile_idx * tile + a_tile_col;
        A_tile[a_tile_row][a_tile_col] = (a_global_row < n && a_global_col < v)? A[a_global_row * v + a_global_col]: 0.0f;

        int b_tile_row = threadIdx.y;
        int b_tile_col = threadIdx.x;
        int b_global_row = tile_idx * tile + b_tile_row;
        int b_global_col = col;
        B_tile[b_tile_row][b_tile_col] = (b_global_row < v && b_global_col < k)? B[b_global_row * k + b_global_col]: 0.0f;

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

//general bias addition
/*
A consists of n*k, bias is B or 1*k
*/
__global__ void matrix_vector_add(float* A, const float* B, int n, int k){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n*k){
        int col = idx%k;
        A[idx]+=B[col];
    }
}

//general relu application: expects a matrix, outputs a matrix
__global__ void kernel_relu(float* A, float* B, int n, int k){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=n*k) return;
    B[idx] = (A[idx]>0.0f) ? A[idx]:0.0f;
}

// general grad relu application: expects a matrix, outputs a matrix
__global__ void kernel_relu_grad(float* grad_in, float* grad_out, float* input, int n, int k){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx>=n*k) return;
    grad_in[idx] = (input[idx]>0.0f)? grad_out[idx]:0.0f;
}

__global__ void maxpool_forward_kernel(const float* input, float* output, int* indices,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int stride, int pad){
    int out_h_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int out_w_idx = threadIdx.y + blockIdx.y*blockDim.y;
    int c = blockIdx.z%channels;
    int b = blockIdx.z/channels;
    if(out_h_idx>=out_h || out_w_idx>=out_w||c>=channels||b>=batch_size) return;

    int out_idx = (((b*channels+c)*out_h)+out_h_idx)*out_w + out_w_idx;

    int in_h_start = out_h_idx*stride;
    int in_w_start = out_w_idx*stride;

    int max_idx = -1;
    float max_val = -1e20f;
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            int in_h_pos = in_h_start+i;
            int in_w_pos = in_w_start +j;
            bool valid = (in_h_pos < in_h) && (in_w_pos < in_w);
            int in_idx = (((b * channels + c) * in_h) + in_h_pos) * in_w + in_w_pos;
            float val = valid?input[in_idx]:-1e20f;
            bool is_max = val>max_val;
            max_val = is_max?val:max_val;
            max_idx = is_max?in_idx:max_idx;
        }
    }
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

__global__ void maxpool_backward_kernel(const float* d_grad_output, float* d_grad_input,
    const int* d_indices, int batch_size, int in_channels, int in_height, int in_width, 
    int out_height, int out_width, int stride, int pad){
    int out_h_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int out_w_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z % in_channels;
    int b = blockIdx.z / in_channels;
    if (out_h_idx >= out_height || out_w_idx >= out_width || c >= in_channels || b >= batch_size) return;

    int out_idx = (((b * in_channels + c) * out_height) + out_h_idx) * out_width + out_w_idx;
    int max_input_idx = d_indices[out_idx];
    float grad_val = d_grad_output[out_idx];

    d_grad_input[max_input_idx] = grad_val;

    }

Linear::Linear(int n, int v, int k): batch_size(n), features(v), num_classes(k) {
    cudaMalloc(&d_weight, sizeof(float)*v*k);
    cudaMalloc(&d_bias, sizeof(float)*k);
    cudaMalloc(&d_input, sizeof(float)*n*v);
    cudaMalloc(&d_before_activation, sizeof(float)*n*k);
    cudaMalloc(&d_output, sizeof(float)*n*k);
    cudaMalloc(&d_weight_grad, sizeof(float)*v*k);
    cudaMalloc(&d_bias_grad, sizeof(float)*k);
    cudaMalloc(&d_grad_input, sizeof(float)*n*v);
    cudaMalloc(&d_grad_before_activation, sizeof(float)*n*k);
    cudaMalloc(&d_grad_output, sizeof(float)*n*k);
}

Linear::~Linear() {
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_input);
    cudaFree(d_before_activation);
    cudaFree(d_output);
    cudaFree(d_weight_grad);
    cudaFree(d_bias_grad);
    cudaFree(d_grad_input);
    cudaFree(d_grad_before_activation);
    cudaFree(d_grad_output);
}

void Linear::forward(const float* input) {
    cudaMemcpy(d_input, input, sizeof(float)*batch_size*features, cudaMemcpyHostToDevice);
    dim3 block(tile,tile);
    dim3 grid((num_classes+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
    matrix_multiply<<<grid, block>>>(d_input, d_weight, d_before_activation, batch_size, features, num_classes);
    dim3 block2(256);
    dim3 grid2((batch_size*num_classes+block2.x-1)/block2.x);
    matrix_vector_add<<<grid2, block2>>>(d_before_activation, d_bias, batch_size, num_classes);
    kernel_relu<<<grid2, block2>>>(d_before_activation, d_output, batch_size, num_classes);
    cudaDeviceSynchronize();
}

void Linear::backward(const float* grad_out) {
    cudaMemcpy(d_grad_output, grad_out, sizeof(float)*batch_size*num_classes, cudaMemcpyHostToDevice);
    //grad_relu<<<grid_size, block_size>>>(d_grad_output, d_grad_before_activation, batch_size, num_classes);
    //matrix_multiply_transpose<<<grid_size, block_size>>>(d_grad_before_activation, d_input, d_weight_grad, batch_size, num_classes, features);
    //matrix_sum<<<grid_size, block_size>>>(d_grad_before_activation, d_bias_grad, batch_size, num_classes);
    //matrix_multiply_transpose<<<grid_size, block_size>>>(d_grad_before_activation, d_weight, d_grad_input, batch_size, num_classes, features);
    cudaDeviceSynchronize();
    //cudaMemcpy(grad_input, d_grad_input, sizeof(float)*batch_size*features, cudaMemcpyDeviceToHost);
}


MaxPool::MaxPool(int n, int depth, int height, int width, int stride, int pad)
    : batch_size(n), in_channels(depth), in_height(height), in_width(width), stride(stride), pad(pad) {
    out_height = (in_height - 2 * pad) / stride;
    out_width = (in_width - 2 * pad) / stride;

    cudaMalloc(&d_input, sizeof(float) * n * height * width * depth);
    cudaMalloc(&d_output, sizeof(float) * n * out_height * out_width * depth);
    cudaMalloc(&d_indices, sizeof(int) * n * out_height * out_width * depth);
    cudaMalloc(&d_grad_input, sizeof(float) * n * height * width * depth);
    cudaMalloc(&d_grad_output, sizeof(float) * n * out_height * out_width * depth);
}

MaxPool::~MaxPool() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_grad_input);
    cudaFree(d_grad_output);
}

void MaxPool::forward(const float* input) {
    cudaMemcpy(d_input, input, sizeof(float) * batch_size * in_height * in_width * in_channels, cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((out_height+block.x-1)/block.x, (out_width+block.y-1)/block.y, batch_size*in_channels);
    maxpool_forward_kernel<<<grid, block>>>(d_input, d_output, d_indices, batch_size, in_channels, in_height, in_width, out_height, out_width, stride, pad);
    cudaDeviceSynchronize();
}

void MaxPool::backward(const float* grad_out) {
    cudaMemcpy(d_grad_output, grad_out, sizeof(float) * batch_size * out_height * out_width * in_channels, cudaMemcpyHostToDevice);
    cudaMemset(d_grad_input, 0, sizeof(float)*batch_size * in_height * in_width * in_channels);
    dim3 block(16,16);
    dim3 grid((out_height+block.x-1)/block.x, (out_width+block.y-1)/block.y, batch_size*in_channels);
    maxpool_backward_kernel<<<grid, block>>>(d_grad_output, d_grad_input, d_indices, batch_size, in_channels, in_height, in_width, out_height, out_width, stride, pad);
    cudaDeviceSynchronize();
    //cudaMemcpy(grad_input, d_grad_input, sizeof(float)*in_height*in_width*in_channels*batch_size, cudaMemcpyDeviceToHost);
}
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include<vector>
#include<algorithm>

struct Linear {
    int batch_size;    // n
    int features;      // v
    int num_classes;   // k

    float* d_weight;   // v*k
    float* d_bias;     // k
    float* d_input;    // n*v
    float* d_before_activation; //n*k
    float* d_output;   // n*k

    float* d_weight_grad; // v*k
    float* d_bias_grad;   // k
    float* d_grad_input;  //n*v
    float* d_grad_before_activation; //n*k
    float* d_grad_output; //n*k

    Linear(int n, int v, int k);
    ~Linear();

    void forward(const float* input);
    void backward(const float* grad_out);
};

struct MaxPool {
    int batch_size;     //n
    int in_channels;    //depth
    int in_height;      //input height
    int in_width;       //input width
    int stride;         //for now same for x and y
    int pad;            // for now no padding will be done
    int out_height;     //output height
    int out_width;      //output width

    float* d_input;     //n*height*width*depth
    float* d_output;    //n*out_height*out_width*depth
    int* d_indices;       //indices to store the max_value for backprop
    float* d_grad_input;
    float* d_grad_output; //n*out_height*out_width*depth

    MaxPool(int n, int depth, int height, int width, int stride, int pad);
    ~MaxPool();

    void forward(const float* input);
    void backward(const float* grad_out);
};

__device__ float relu(float x);      //relu function
__device__ float relu_grad(float x); //relu gradient


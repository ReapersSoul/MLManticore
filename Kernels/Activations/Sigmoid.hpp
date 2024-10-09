#pragma once
#include <string>

static std::string Sigmoid_Kernel_Code = R"(


__kernel void Sigmoid(__global float* input, __global float* output, const int size) {
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;
    for(int j=start; j<end; j++) {
        output[j] = input[j];//1.0f / (1.0f + exp(-input[j]));
    }
}
)";
#pragma once
#include <string>

static std::string Tanh_Kernel_Code = R"(


__kernel void Tanh(__global const float* tensor, __global float* result, int size)
{
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;

    for(int j=start; j<end; j++) {
        result[j] = tanh(tensor[j]);
    }
}
)";
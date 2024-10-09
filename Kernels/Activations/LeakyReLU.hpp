#pragma once
#include <string>

static std::string LeakyReLU_Kernel_Code = R"(


__kernel void LeakyReLU(__global const float* tensor, __global float* result, int size, float alpha)
{
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;

    for(int j=start; j<end; j++) {
        if(tensor[j] > 0)
        {
            result[j] = tensor[j];
        }
        else
        {
            result[j] = alpha * tensor[j];
        }
    }
}
)";
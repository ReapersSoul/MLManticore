#pragma once
#include <string>

static std::string LeakyReLU_Derivative_Kernel_Code = R"(


__kernel void LeakyReLU_Derivative(__global const float* tensor, __global float* result, int size, float alpha)
{
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;

    for(int j=start; j<end; j++) {
        if(tensor[j] > 0)
        {
            result[j] = 1;
        }
        else
        {
            result[j] = alpha;
        }
    }
}
)";
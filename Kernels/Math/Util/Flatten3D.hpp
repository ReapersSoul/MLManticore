#pragma once

#include <string>

static std::string Flatten3D_Kernel_Code = R"(
__kernel void Flatten3D(__global const float* input, __global float* output, int size)
{
    int i = get_global_id(0);
    int threads = get_global_size(0);
    int batch_size = size / threads;
    int start = i * batch_size;
    int end = start + batch_size;
    if (i == threads - 1)
    {
        end = size;
    }
    for (int j = start; j < end; j++)
    {
        output[j] = input[j];
    }
}
)";
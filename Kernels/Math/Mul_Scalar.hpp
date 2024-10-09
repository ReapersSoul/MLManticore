#pragma once
#include <string>

static std::string Mul_Scalar_Kernel_Code = R"(


__kernel void Mul_Scalar(__global const float* input1,
                         const float input2,
                         __global float* output,
                         const int size)
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
        output[j] = input1[j] * input2;
    }
}
)";
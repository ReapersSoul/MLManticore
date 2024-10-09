#pragma once
#include <string>

static std::string Add_Kernel_Code = R"(


__kernel void Add(
    __global float* A,
    __global float* B,
    __global float* C,
    const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
    {
        C[i] = A[i] + B[i];
    }
}
)";
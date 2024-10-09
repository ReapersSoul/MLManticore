#pragma once
#include <string>

static std::string Matrix_Transpose_Kernel_Code = R"(


__kernel void Matrix_Transpose(__global float* inputMatrix,
                               __global float* outputMatrix,
                               const int rows,
                               const int cols)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    outputMatrix[j * rows + i] = inputMatrix[i * cols + j];
}
)";
#pragma once

#include <string>

static std::string Reshape3D_Kernel_Code = R"(
__kernel void Reshape3D(__global const float* input, __global float* output, int size, int rows, int cols, int depth)
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
        int row = j / (cols * depth);
        int col = (j / depth) % cols;
        int d = j % depth;
        int index = row * cols * depth + col * depth + d;
        output[index] = input[j];
    }
}
)";
#pragma once
#include <string>

static std::string SoftMax_Kernel_Code = R"(


__kernel void SoftMax(__global const float *input, __global float *output, const int size,
                      const float max_val, const float sum) {
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;

    for(int j=start; j<end; j++) {
        output[j] = exp(input[j] - max_val) / sum;
    }
};
)";
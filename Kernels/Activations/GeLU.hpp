#pragma once
#include <string>

static std::string GeLU_Kernel_Code = R"(


__kernel void GeLU(__global float* input, __global float* output, const int size) {
    int i = get_global_id(0);
    int threads= get_global_size(0);
    int batch_size = size/threads;
    int start = i*batch_size;
    int end = start + batch_size;

    for(int j=start; j<end; j++) {
        output[j] = 0.5 * input[j] * (1.0 + tanh(0.7978845608 * (input[j] + 0.044715 * input[j] * input[j] * input[j])));
    }
}
)";
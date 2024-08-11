__kernel void Sigmoid(__global double* input, __global double* output, const int size) {
	int i = get_global_id(0);
	int threads= get_global_size(0);
	int batch_size = size/threads;
	int start = i*batch_size;
	int end = start + batch_size;
	for(int j=start; j<end; j++) {
		output[j] = input[j];//1.0f / (1.0f + exp(-input[j]));
	}
}
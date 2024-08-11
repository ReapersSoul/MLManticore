__kernel void Sigmoid_Derivative(__global double* tensor, __global double* result, const int size)
{
	int i = get_global_id(0);
	int threads= get_global_size(0);
	int batch_size = size/threads;
	int start = i*batch_size;
	int end = start + batch_size;

	for(int j=start; j<end; j++) {
		result[j] = tensor[j] * (1 - tensor[j]);
	}
}
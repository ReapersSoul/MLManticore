__kernel void GeLU_Derivative(__global const double* tensor, __global double* result, int size)
{
	int i = get_global_id(0);
	int threads= get_global_size(0);
	int batch_size = size/threads;
	int start = i*batch_size;
	int end = start + batch_size;

	for(int j=start; j<end; j++) {
		result[j] = 0.5 * (1 + erf(tensor[j] / sqrt(2.0)));	
	}
}
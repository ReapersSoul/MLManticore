__kernel void Sub(__global const double* input1,
							  __global const double* input2,
							  __global double* output,
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
		output[j] = input1[j] - input2[j];
	}
}
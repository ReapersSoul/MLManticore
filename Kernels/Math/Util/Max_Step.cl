__kernel void Max_Step(__global const double* input, __global double* output, int size)
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
	for (int j = start; j < end; j+=2)
	{
		if (input[j] > input[j + 1])
		{
			output[j / 2] = input[j];
		}
		else
		{
			output[j / 2] = input[j + 1];
		}
	}
}
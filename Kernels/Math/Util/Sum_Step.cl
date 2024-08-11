__kernel void Sum_Step(__global const double* input, __global double* output, int size)
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
		output[i] += input[j] + input[j + 1];
	}
}

double lerp(double a, double b, double t)
{
	return a + (b - a) * t;
}
// static void GPU_LeakyReLU(std::vector<double> tensor, std::vector<double> &result, double alpha)
// {
// 	// check if OpenCL is initialized
// 	if (!isInitialized)
// 	{
// 		InitializeOpenCL();
// 	}

// 	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
// 	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));

// 	cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(double));

// 	int size = tensor.size();

// 	// write the data to the buffers
// 	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());

// 	queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(double), &alpha);

// 	// set the arguments
// 	LeakyReLU_Kernel.setArg(0, tensorBuffer);
// 	LeakyReLU_Kernel.setArg(1, resultBuffer);
// 	LeakyReLU_Kernel.setArg(2, size);
// 	LeakyReLU_Kernel.setArg(3, alphaBuffer);

// 	// get the max work group size
// 	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
// 	if (maxWorkGroupSize > size)
// 	{
// 		maxWorkGroupSize = size;
// 	}

// 	// execute the kernel
// 	queue.enqueueNDRangeKernel(LeakyReLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

// 	// read the result
// 	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
// }

__kernel void LeakyReLU(__global const double* tensor, __global double* result, int size, double alpha)
{
	int i = get_global_id(0);
	int threads= get_global_size(0);
	int batch_size = size/threads;
	int start = i*batch_size;
	int end = start + batch_size;

	for(int j=start; j<end; j++) {
		if(tensor[j] > 0)
		{
			result[j] = tensor[j];
		}
		else
		{
			result[j] = alpha * tensor[j];
		}
	}
}
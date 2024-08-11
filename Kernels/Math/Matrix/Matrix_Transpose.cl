// static void GPU_TransposeMatrix(std::vector<std::vector<double>> Matrix, std::vector<std::vector<double>> &Result)
// {
// 	// check if OpenCL is initialized
// 	if (!isInitialized)
// 	{
// 		InitializeOpenCL();
// 	}

// 	int Rows = Matrix.size();
// 	int Cols = Matrix[0].size();

// 	// create the buffers
// 	cl::Buffer matrixBuffer(context, CL_MEM_READ_ONLY, Rows * Cols * sizeof(double));
// 	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Rows * Cols * sizeof(double));
// 	cl::Buffer rowsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
// 	cl::Buffer colsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

// 	// write the data to the buffers
// 	queue.enqueueWriteBuffer(matrixBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), Matrix.data());
// 	queue.enqueueWriteBuffer(rowsBuffer, CL_TRUE, 0, sizeof(int), &Rows);
// 	queue.enqueueWriteBuffer(colsBuffer, CL_TRUE, 0, sizeof(int), &Cols);

// 	// set the arguments
// 	Matrix_Transpose_Kernel.setArg(0, matrixBuffer);
// 	Matrix_Transpose_Kernel.setArg(1, resultBuffer);
// 	Matrix_Transpose_Kernel.setArg(2, rowsBuffer);
// 	Matrix_Transpose_Kernel.setArg(3, colsBuffer);

// 	// execute the kernel
// 	queue.enqueueNDRangeKernel(Matrix_Transpose_Kernel, cl::NullRange, cl::NDRange(Rows, Cols), cl::NullRange);

// 	// read the result
// 	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), Result.data());
// }

__kernel void Matrix_Transpose(__global double* inputMatrix,
							  __global double* outputMatrix,
							  const int rows,
							  const int cols)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	outputMatrix[j * rows + i] = inputMatrix[i * cols + j];
}
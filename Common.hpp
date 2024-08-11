#pragma once
#define TRACY_ENABLE
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <filesystem>
#include <CL/cl2.hpp>
#include <tracy/Tracy.hpp>
#include <Tensor/Tensor.hpp>

template <typename T>
static T RandRange(T min, T max)
{
	ZoneScoped;
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<T> dis(min, max);
	return dis(gen);
}

// function to read the file into a string

static std::string slurp(std::ifstream &in)
{
	ZoneScoped;
	std::ostringstream sstr;
	sstr << in.rdbuf();
	return sstr.str();
}

// OpenCL variables
static std::vector<cl::Platform> platforms;
static std::vector<cl::Device> devices;
static cl::Context context;
static cl::CommandQueue queue;
static bool isInitialized = false;
static int maxWorkGroupSize;

// function to print the build error

static void PrintClBuildError(std::string KernelName, cl_int err, cl::Program &Program, cl::Device &Device, int line)
{
	ZoneScoped;
	char *buff_erro;
	cl_int errcode;
	size_t build_log_len;
	errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	if (errcode)
	{
		printf("clGetProgramBuildInfo failed at line %d\n", line);
		throw std::runtime_error("clGetProgramBuildInfo failed");
	}

	buff_erro = (char *)malloc(build_log_len);
	if (!buff_erro)
	{
		printf("malloc failed at line %d\n", line);
		throw std::runtime_error("malloc failed");
	}

	errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
	if (errcode)
	{
		printf("clGetProgramBuildInfo failed at line %d\n", line);
		throw std::runtime_error("clGetProgramBuildInfo failed");
	}

	fprintf(stderr, "Build log for %s:\n%s\n", KernelName.c_str(), buff_erro); // Be careful with  the fprint
	free(buff_erro);
	fprintf(stderr, "clBuildProgram failed\n");
}

static const char *getErrorString(cl_int error)
{
	ZoneScoped;
	switch (error)
	{
	// run-time and JIT compiler errors
	case 0:
		return "CL_SUCCESS";
	case -1:
		return "CL_DEVICE_NOT_FOUND";
	case -2:
		return "CL_DEVICE_NOT_AVAILABLE";
	case -3:
		return "CL_COMPILER_NOT_AVAILABLE";
	case -4:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5:
		return "CL_OUT_OF_RESOURCES";
	case -6:
		return "CL_OUT_OF_HOST_MEMORY";
	case -7:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8:
		return "CL_MEM_COPY_OVERLAP";
	case -9:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case -10:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11:
		return "CL_BUILD_PROGRAM_FAILURE";
	case -12:
		return "CL_MAP_FAILURE";
	case -13:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case -16:
		return "CL_LINKER_NOT_AVAILABLE";
	case -17:
		return "CL_LINK_PROGRAM_FAILURE";
	case -18:
		return "CL_DEVICE_PARTITION_FAILED";
	case -19:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	// compile-time errors
	case -30:
		return "CL_INVALID_VALUE";
	case -31:
		return "CL_INVALID_DEVICE_TYPE";
	case -32:
		return "CL_INVALID_PLATFORM";
	case -33:
		return "CL_INVALID_DEVICE";
	case -34:
		return "CL_INVALID_CONTEXT";
	case -35:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case -36:
		return "CL_INVALID_COMMAND_QUEUE";
	case -37:
		return "CL_INVALID_HOST_PTR";
	case -38:
		return "CL_INVALID_MEM_OBJECT";
	case -39:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40:
		return "CL_INVALID_IMAGE_SIZE";
	case -41:
		return "CL_INVALID_SAMPLER";
	case -42:
		return "CL_INVALID_BINARY";
	case -43:
		return "CL_INVALID_BUILD_OPTIONS";
	case -44:
		return "CL_INVALID_PROGRAM";
	case -45:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46:
		return "CL_INVALID_KERNEL_NAME";
	case -47:
		return "CL_INVALID_KERNEL_DEFINITION";
	case -48:
		return "CL_INVALID_KERNEL";
	case -49:
		return "CL_INVALID_ARG_INDEX";
	case -50:
		return "CL_INVALID_ARG_VALUE";
	case -51:
		return "CL_INVALID_ARG_SIZE";
	case -52:
		return "CL_INVALID_KERNEL_ARGS";
	case -53:
		return "CL_INVALID_WORK_DIMENSION";
	case -54:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case -55:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case -56:
		return "CL_INVALID_GLOBAL_OFFSET";
	case -57:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case -58:
		return "CL_INVALID_EVENT";
	case -59:
		return "CL_INVALID_OPERATION";
	case -60:
		return "CL_INVALID_GL_OBJECT";
	case -61:
		return "CL_INVALID_BUFFER_SIZE";
	case -62:
		return "CL_INVALID_MIP_LEVEL";
	case -63:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64:
		return "CL_INVALID_PROPERTY";
	case -65:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66:
		return "CL_INVALID_COMPILER_OPTIONS";
	case -67:
		return "CL_INVALID_LINKER_OPTIONS";
	case -68:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";

	// extension errors
	case -1000:
		return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001:
		return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002:
		return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003:
		return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004:
		return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005:
		return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default:
		return "Unknown OpenCL error";
	}
}

static void PrintClError(std::string KernelName, cl_int err, int line)
{
	ZoneScoped;
	printf("Error Loading %s at line %d: %s\n", KernelName.c_str(), line, getErrorString(err));
}

// kernels and programs
static cl::Kernel Sum_Step_Kernel;
static cl::Program Sum_Step_Program;

static cl::Kernel Mul_Kernel;
static cl::Program Mul_Program;

static cl::Kernel Mul_Scalar_Kernel;
static cl::Program Mul_Scalar_Program;

static cl::Kernel Sub_Kernel;
static cl::Program Sub_Program;

static cl::Kernel Add_Kernel;
static cl::Program Add_Program;

static cl::Kernel Div_Kernel;
static cl::Program Div_Program;

static cl::Kernel Abs_Kernel;
static cl::Program Abs_Program;

static cl::Kernel Abs_Derivative_Kernel;
static cl::Program Abs_Derivative_Program;

// matrix kernels and programs
static cl::Kernel Matrix_Transpose_Kernel;
static cl::Program Matrix_Transpose_Program;

// tensor kernels and programs
static cl::Kernel Tensor_Block_Kernel;
static cl::Program Tensor_Block_Program;

static cl::Kernel Tensor_Transpose_Kernel;
static cl::Program Tensor_Transpose_Program;

static cl::Kernel Tensor_Dot_Kernel;
static cl::Program Tensor_Dot_Program;

static cl::Kernel Tensor_Conv_Kernel;
static cl::Program Tensor_Conv_Program;

// activation functions
static cl::Kernel Sigmoid_Kernel;
static cl::Program Sigmoid_Program;

static cl::Kernel ReLU_Kernel;
static cl::Program ReLU_Program;

static cl::Kernel LeakyReLU_Kernel;
static cl::Program LeakyReLU_Program;

static cl::Kernel Tanh_Kernel;
static cl::Program Tanh_Program;

static cl::Kernel Softmax_Kernel;
static cl::Program Softmax_Program;

static cl::Kernel GeLU_Kernel;
static cl::Program GeLU_Program;

// activation derivatives

static cl::Kernel Sigmoid_Derivative_Kernel;
static cl::Program Sigmoid_Derivative_Program;

static cl::Kernel ReLU_Derivative_Kernel;
static cl::Program ReLU_Derivative_Program;

static cl::Kernel LeakyReLU_Derivative_Kernel;
static cl::Program LeakyReLU_Derivative_Program;

static cl::Kernel Tanh_Derivative_Kernel;
static cl::Program Tanh_Derivative_Program;

static cl::Kernel Softmax_Derivative_Kernel;
static cl::Program Softmax_Derivative_Program;

static cl::Kernel GeLU_Derivative_Kernel;
static cl::Program GeLU_Derivative_Program;

// Util
static cl::Kernel Max_Step_Kernel;
static cl::Program Max_Step_Program;

// function to initialize OpenCL and create the kernels and programs

static void MakeKernel(
	std::string kernel_name,
	std::string kernel_path,
	cl::Kernel &kernel,
	cl::Program &program,
	int line)
{
	ZoneScoped;
	std::ifstream file(kernel_path);
	if (!file.is_open())
	{
		printf("Error Loading %s at line %d: Could not open file\n", kernel_name.c_str(), line);
	}
	std::string kernel_code = slurp(file);
	file.close();
	cl::Program::Sources sources;
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	program = cl::Program(context, sources);
	cl_int err = program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError(kernel_name, err, program, devices[0], line);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	kernel = cl::Kernel(program, kernel_name.c_str(), &err);
	if (err != CL_SUCCESS)
	{
		PrintClError(kernel_name, err, line);
		throw std::runtime_error("Could not create kernel: " + kernel_name);
	}
}

static void InitializeOpenCL()
{
	ZoneScoped;
	if (isInitialized)
	{
		return;
	}
	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		throw std::runtime_error("No OpenCL platforms found!");
	}

	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.empty())
	{
		throw std::runtime_error("No GPU devices found!");
	}

	// get the max work group GetSize
	maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

	char ext_str[8192];
	size_t ext_str_len;

	// setup context and queue for printing to console
	context = cl::Context(devices);

	queue = cl::CommandQueue(context, devices[0]);

	// get device extensions
	clGetDeviceInfo(devices[0](), CL_DEVICE_EXTENSIONS, 8192, ext_str, &ext_str_len);
	// printf("Device extensions: %s\n", ext_str);

	// vector kernels
	MakeKernel("Sum_Step", "./MLLib/Kernels/Math/Util/Sum_Step.cl", Sum_Step_Kernel, Sum_Step_Program, __LINE__);

	MakeKernel("Mul", "./MLLib/Kernels/Math/Mul.cl", Mul_Kernel, Mul_Program, __LINE__);

	MakeKernel("Mul_Scalar", "./MLLib/Kernels/Math/Mul_Scalar.cl", Mul_Scalar_Kernel, Mul_Scalar_Program, __LINE__);

	MakeKernel("Sub", "./MLLib/Kernels/Math/Sub.cl", Sub_Kernel, Sub_Program, __LINE__);

	MakeKernel("Add", "./MLLib/Kernels/Math/Add.cl", Add_Kernel, Add_Program, __LINE__);

	MakeKernel("Div", "./MLLib/Kernels/Math/Div.cl", Div_Kernel, Div_Program, __LINE__);

	MakeKernel("Abs", "./MLLib/Kernels/Math/Abs.cl", Abs_Kernel, Abs_Program, __LINE__);

	MakeKernel("Abs_Derivative", "./MLLib/Kernels/Math/Abs_Derivative.cl", Abs_Derivative_Kernel, Abs_Derivative_Program, __LINE__);

	// matrix kernels
	MakeKernel("Matrix_Transpose", "./MLLib/Kernels/Math/matrix/Matrix_Transpose.cl", Matrix_Transpose_Kernel, Matrix_Transpose_Program, __LINE__);

	// tensor kernels
	// MakeKernel("Tensor_Block", "./MLLib/Kernels/Math/tensor/Tensor_Block.cl", Tensor_Block_Kernel, Tensor_Block_Program,__LINE__);

	// MakeKernel("Tensor_Transpose", "./MLLib/Kernels/Math/tensor/Tensor_Transpose.cl", Tensor_Transpose_Kernel, Tensor_Transpose_Program,__LINE__);

	// MakeKernel("Tensor_Dot", "./MLLib/Kernels/Math/tensor/Tensor_Dot.cl", Tensor_Dot_Kernel, Tensor_Dot_Program,__LINE__);

	// MakeKernel("Tensor_Conv", "./MLLib/Kernels/Math/tensor/Tensor_Conv.cl", Tensor_Conv_Kernel, Tensor_Conv_Program,__LINE__);

	// activation functions
	MakeKernel("Sigmoid", "./MLLib/Kernels/Activations/Sigmoid.cl", Sigmoid_Kernel, Sigmoid_Program, __LINE__);

	MakeKernel("ReLU", "./MLLib/Kernels/Activations/ReLU.cl", ReLU_Kernel, ReLU_Program, __LINE__);

	MakeKernel("LeakyReLU", "./MLLib/Kernels/Activations/LeakyReLU.cl", LeakyReLU_Kernel, LeakyReLU_Program, __LINE__);

	MakeKernel("Tanh", "./MLLib/Kernels/Activations/Tanh.cl", Tanh_Kernel, Tanh_Program, __LINE__);

	MakeKernel("SoftMax", "./MLLib/Kernels/Activations/SoftMax.cl", Softmax_Kernel, Softmax_Program, __LINE__);

	MakeKernel("GeLU", "./MLLib/Kernels/Activations/GeLU.cl", GeLU_Kernel, GeLU_Program, __LINE__);

	// activation derivatives
	MakeKernel("Sigmoid_Derivative", "./MLLib/Kernels/Activation_Derivatives/Sigmoid_Derivative.cl", Sigmoid_Derivative_Kernel, Sigmoid_Derivative_Program, __LINE__);

	MakeKernel("ReLU_Derivative", "./MLLib/Kernels/Activation_Derivatives/ReLU_Derivative.cl", ReLU_Derivative_Kernel, ReLU_Derivative_Program, __LINE__);

	MakeKernel("LeakyReLU_Derivative", "./MLLib/Kernels/Activation_Derivatives/LeakyReLU_Derivative.cl", LeakyReLU_Derivative_Kernel, LeakyReLU_Derivative_Program, __LINE__);

	MakeKernel("Tanh_Derivative", "./MLLib/Kernels/Activation_Derivatives/Tanh_Derivative.cl", Tanh_Derivative_Kernel, Tanh_Derivative_Program, __LINE__);

	MakeKernel("SoftMax_Derivative", "./MLLib/Kernels/Activation_Derivatives/SoftMax_Derivative.cl", Softmax_Derivative_Kernel, Softmax_Derivative_Program, __LINE__);

	MakeKernel("GeLU_Derivative", "./MLLib/Kernels/Activation_Derivatives/GeLU_Derivative.cl", GeLU_Derivative_Kernel, GeLU_Derivative_Program, __LINE__);

	// Util
	MakeKernel("Max_Step", "./MLLib/Kernels/Math/Util/Max_Step.cl", Max_Step_Kernel, Max_Step_Program, __LINE__);

	// set the flag
	isInitialized = true;
}

// vector functions
static void GPU_Sum_Step(Vector<double> data, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	result.resize(data.GetSize() / 2);

	// create the buffers
	cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, data.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (data.GetSize() / 2) * sizeof(double));

	int GetSize = data.GetSize();

	// write the data to the buffers
	cl_int err;
	err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.GetSize() * sizeof(double), data.GetData());
	if (err != CL_SUCCESS)
	{
		PrintClError("dataBuffer", err, __LINE__);
		throw std::runtime_error("Could not write to dataBuffer");
	}

	// set the arguments
	err = Sum_Step_Kernel.setArg(0, dataBuffer);
	if (err != CL_SUCCESS)
	{
		PrintClError("dataBuffer", err, __LINE__);
		throw std::runtime_error("Could not set dataBuffer");
	}
	err = Sum_Step_Kernel.setArg(1, resultBuffer);
	if (err != CL_SUCCESS)
	{
		PrintClError("resultBuffer", err, __LINE__);
		throw std::runtime_error("Could not set resultBuffer");
	}
	err = Sum_Step_Kernel.setArg(2, GetSize);
	if (err != CL_SUCCESS)
	{
		PrintClError("GetSize", err, __LINE__);
		throw std::runtime_error("Could not set GetSize");
	}

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > (GetSize / 2))
	{
		maxWorkGroupSize = (GetSize / 2);
	}

	// execute the kernel
	err = queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);
	if (err != CL_SUCCESS)
	{
		PrintClError("Sum_Step_Kernel", err, __LINE__);
		throw std::runtime_error("Could not execute the kernel");
	}

	// resize the result vector
	result.resize(GetSize / 2);

	// wait for the kernel to finish executing
	queue.finish();

	// read the result
	err = queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (data.GetSize() / 2) * sizeof(double), result.GetData());
	if (err != CL_SUCCESS)
	{
		PrintClError("resultBuffer", err, __LINE__);
		throw std::runtime_error("Could not read from resultBuffer");
	}

	// if the result is not a power of 2, add the last element
	if (data.GetSize() % 2 != 0)
	{
		int size = data.GetSize();
		std::array<int,1> index{size - 1};
		result.push_back(data[index]);
	}
}

static void GPU_Sum(Vector<double> data, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp = data;
	while (temp.GetSize() > 1)
	{
		Vector<double> temp2;
		temp2.resize(temp.GetSize() / 2);
		GPU_Sum_Step(temp, temp2);
		temp = temp2;
	}
	result = temp[std::array<int,1>{0}];
}

static void GPU_Sum_Fast(Vector<double> data, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	// create the buffers
	cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, data.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (data.GetSize() / 2) * sizeof(double));

	int GetSize = data.GetSize();
	int times = data.GetSize() / 2;

	for (int i = 0; i < times; i++)
	{
		// write the data to the buffers
		cl_int err;
		err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.GetSize() * sizeof(double), data.GetData());
		if (err != CL_SUCCESS)
		{
			PrintClError("dataBuffer", err, __LINE__);
			throw std::runtime_error("Could not write to dataBuffer");
		}

		// set the arguments
		err = Sum_Step_Kernel.setArg(0, dataBuffer);
		if (err != CL_SUCCESS)
		{
			PrintClError("dataBuffer", err, __LINE__);
			throw std::runtime_error("Could not set dataBuffer");
		}
		err = Sum_Step_Kernel.setArg(1, resultBuffer);
		if (err != CL_SUCCESS)
		{
			PrintClError("resultBuffer", err, __LINE__);
			throw std::runtime_error("Could not set resultBuffer");
		}
		err = Sum_Step_Kernel.setArg(2, GetSize);
		if (err != CL_SUCCESS)
		{
			PrintClError("GetSize", err, __LINE__);
			throw std::runtime_error("Could not set GetSize");
		}

		// get the max work group GetSize
		size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		if (maxWorkGroupSize > (GetSize / 2))
		{
			maxWorkGroupSize = (GetSize / 2);
		}

		// execute the kernel
		err = queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);
		if (err != CL_SUCCESS)
		{
			PrintClError("Sum_Step_Kernel", err, __LINE__);
			throw std::runtime_error("Could not execute the kernel");
		}

		GetSize = GetSize / 2;
	}

	// wait for the kernel to finish executing
	queue.finish();

	Vector<double> temp;
	temp.resize(data.GetSize() / 2);

	// read the result
	cl_int err = queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (data.GetSize() / 2) * sizeof(double), temp.GetData());
	if (err != CL_SUCCESS)
	{
		PrintClError("resultBuffer", err, __LINE__);
		throw std::runtime_error("Could not read from resultBuffer");
	}

	result = temp[std::array<int,1>{0}];

	// if the result is not a power of 2, add the last element
	if (data.GetSize() % 2 != 0)
	{
		result+=data[std::array<int,1>{data.GetSize() - 1}];
	}
}

static void GPU_Mul(Vector<double> data1, Vector<double> data2, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	// check if the GetSize of the data1 is equal to the GetSize of the data2
	if (data1.GetSize() != data2.GetSize())
	{
		throw std::invalid_argument("The GetSize of the data1 is not equal to the GetSize of the data2");
	}

	int GetSize = data1.GetSize();

	// create the buffers
	cl::Buffer data1Buffer(context, CL_MEM_READ_ONLY, GetSize * sizeof(double));
	cl::Buffer data2Buffer(context, CL_MEM_READ_ONLY, GetSize * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, GetSize * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(data1Buffer, CL_TRUE, 0, GetSize * sizeof(double), data1.GetData());
	queue.enqueueWriteBuffer(data2Buffer, CL_TRUE, 0, GetSize * sizeof(double), data2.GetData());

	// set the arguments
	Mul_Kernel.setArg(0, data1Buffer);
	Mul_Kernel.setArg(1, data2Buffer);
	Mul_Kernel.setArg(2, resultBuffer);
	Mul_Kernel.setArg(3, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Mul_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// resize the result vector
	result.resize(GetSize);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_MulSum(Vector<double> tensor1, Vector<double> tensor2, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(tensor1.GetSize());
	GPU_Mul(tensor1, tensor2, temp);
	GPU_Sum_Fast(temp, result);
}

static void GPU_Mul_Scalar(Vector<double> tensor, double scalar, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Mul_Scalar_Kernel.setArg(0, tensorBuffer);
	Mul_Scalar_Kernel.setArg(1, scalar);
	Mul_Scalar_Kernel.setArg(2, resultBuffer);
	Mul_Scalar_Kernel.setArg(3, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Mul_Scalar_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// resize the result vector
	result.resize(GetSize);

	queue.finish();

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Sub(Vector<double> tensor1, Vector<double> tensor2, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.GetSize() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.GetSize() * sizeof(double));

	int GetSize = tensor1.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.GetSize() * sizeof(double), tensor1.GetData());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.GetSize() * sizeof(double), tensor2.GetData());

	// set the arguments
	Sub_Kernel.setArg(0, tensor1Buffer);
	Sub_Kernel.setArg(1, tensor2Buffer);
	Sub_Kernel.setArg(2, resultBuffer);
	Sub_Kernel.setArg(3, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Sub_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// resize the result vector
	result.resize(GetSize);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Add(Vector<double> tensor1, Vector<double> tensor2, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.GetSize() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.GetSize() * sizeof(double));

	int GetSize = tensor1.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.GetSize() * sizeof(double), tensor1.GetData());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.GetSize() * sizeof(double), tensor2.GetData());

	// set the arguments
	Add_Kernel.setArg(0, tensor1Buffer);
	Add_Kernel.setArg(1, tensor2Buffer);
	Add_Kernel.setArg(2, resultBuffer);
	Add_Kernel.setArg(3, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Add_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// resize the result vector
	result.resize(GetSize);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Div(Vector<double> tensor1, Vector<double> tensor2, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.GetSize() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.GetSize() * sizeof(double));

	int GetSize = tensor1.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.GetSize() * sizeof(double), tensor1.GetData());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.GetSize() * sizeof(double), tensor2.GetData());

	// set the arguments
	Div_Kernel.setArg(0, tensor1Buffer);
	Div_Kernel.setArg(1, tensor2Buffer);
	Div_Kernel.setArg(2, resultBuffer);
	Div_Kernel.setArg(3, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Div_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// resize the result vector
	result.resize(GetSize);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Abs(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Abs_Kernel.setArg(0, tensorBuffer);
	Abs_Kernel.setArg(1, resultBuffer);
	Abs_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Abs_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Abs_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Abs_Derivative_Kernel.setArg(0, tensorBuffer);
	Abs_Derivative_Kernel.setArg(1, resultBuffer);
	Abs_Derivative_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Abs_Derivative_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

// matrix functions
static void GPU_TransposeMatrix(Matrix<double> matrix, Matrix<double> &Result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	int Rows = matrix.GetShape()[0];
	int Cols = matrix.GetShape()[1];

	Vector<double> _Result(Rows * Cols);
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < Cols; j++)
		{
			_Result[std::array<int,1>{j * Rows + i}] = matrix[std::array<int,2>{i,j}];
		}
	}

	// create the buffers
	cl::Buffer matrixBuffer(context, CL_MEM_READ_ONLY, Rows * Cols * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Rows * Cols * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(matrixBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), _Result.GetData());

	// set the arguments
	Matrix_Transpose_Kernel.setArg(0, matrixBuffer);
	Matrix_Transpose_Kernel.setArg(1, resultBuffer);
	Matrix_Transpose_Kernel.setArg(2, Rows);
	Matrix_Transpose_Kernel.setArg(3, Cols);

	// execute the kernel
	queue.enqueueNDRangeKernel(Matrix_Transpose_Kernel, cl::NullRange, cl::NDRange(Rows, Cols), cl::NullRange);

	queue.finish();

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), _Result.GetData());

	Result.resize(Cols, Rows);
	for (int i = 0; i < Cols; i++)
	{
		for (int j = 0; j < Rows; j++)
		{
			Result[std::array<int,2>{i,j}] = _Result[std::array<int,1>{i * Rows + j}];
		}
	}
}

// tensor functions
static void Tensor_Block(Tensor<double> , std::vector<int> BlockStart, std::vector<int> BlockShape, Vector<double> &Result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	int TensorDims = TensorShape.size();
	int BlockDims = BlockShape.size();

	// create the buffers
	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer tensorShapeBuffer(context, CL_MEM_READ_ONLY, TensorShape.GetSize() * sizeof(int));
	cl::Buffer blockStartBuffer(context, CL_MEM_READ_WRITE, BlockStart.GetSize() * sizeof(int));
	cl::Buffer tensorDimsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	cl::Buffer blockShapeBuffer(context, CL_MEM_READ_ONLY, BlockShape.GetSize() * sizeof(int));
	cl::Buffer blockDimsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Result.GetSize() * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());
	queue.enqueueWriteBuffer(tensorShapeBuffer, CL_TRUE, 0, TensorShape.GetSize() * sizeof(int), TensorShape.GetData());
	queue.enqueueWriteBuffer(blockStartBuffer, CL_TRUE, 0, BlockStart.GetSize() * sizeof(int), BlockStart.GetData());
	queue.enqueueWriteBuffer(tensorDimsBuffer, CL_TRUE, 0, sizeof(int), &TensorDims);

	queue.enqueueWriteBuffer(blockShapeBuffer, CL_TRUE, 0, BlockShape.GetSize() * sizeof(int), BlockShape.GetData());
	queue.enqueueWriteBuffer(blockDimsBuffer, CL_TRUE, 0, sizeof(int), &BlockDims);

	// set the arguments
	Tensor_Block_Kernel.setArg(0, tensorBuffer);
	Tensor_Block_Kernel.setArg(1, tensorShapeBuffer);
	Tensor_Block_Kernel.setArg(2, blockStartBuffer);
	Tensor_Block_Kernel.setArg(3, tensorDimsBuffer);

	Tensor_Block_Kernel.setArg(4, blockShapeBuffer);
	Tensor_Block_Kernel.setArg(5, blockDimsBuffer);

	Tensor_Block_Kernel.setArg(6, resultBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Tensor_Block_Kernel, cl::NullRange, cl::NDRange(Result.GetSize()), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Result.GetSize() * sizeof(double), Result.GetData());
}

// Activation functions
static void GPU_Sigmoid(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Sigmoid_Kernel.setArg(0, tensorBuffer);
	Sigmoid_Kernel.setArg(1, resultBuffer);
	Sigmoid_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Sigmoid_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_ReLU(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	ReLU_Kernel.setArg(0, tensorBuffer);
	ReLU_Kernel.setArg(1, resultBuffer);
	ReLU_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(ReLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_LeakyReLU(Vector<double> tensor, Vector<double> &result, double alpha)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(double), &alpha);

	// set the arguments
	LeakyReLU_Kernel.setArg(0, tensorBuffer);
	LeakyReLU_Kernel.setArg(1, resultBuffer);
	LeakyReLU_Kernel.setArg(2, GetSize);
	LeakyReLU_Kernel.setArg(3, alphaBuffer);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(LeakyReLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Tanh(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Tanh_Kernel.setArg(0, tensorBuffer);
	Tanh_Kernel.setArg(1, resultBuffer);
	Tanh_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Tanh_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_SoftMax(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Softmax_Kernel.setArg(0, tensorBuffer);
	Softmax_Kernel.setArg(1, resultBuffer);
	Softmax_Kernel.setArg(2, GetSize);

	float max_val = tensor[0];
	for (int i = 1; i < tensor.GetSize(); ++i)
	{
		if (tensor[i] > max_val)
		{
			max_val = tensor[i];
		}
	}

	float sum = 0.0f;
	for (int i = 0; i < tensor.GetSize(); ++i)
	{
		sum += exp(tensor[i] - max_val);
	}

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Softmax_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_GeLU(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	GeLU_Kernel.setArg(0, tensorBuffer);
	GeLU_Kernel.setArg(1, resultBuffer);
	GeLU_Kernel.setArg(2, GetSize);

	// get the max work group GetSize
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if (maxWorkGroupSize > GetSize)
	{
		maxWorkGroupSize = GetSize;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(GeLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

// Activation derivatives
static void GPU_Sigmoid_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Sigmoid_Derivative_Kernel.setArg(0, tensorBuffer);
	Sigmoid_Derivative_Kernel.setArg(1, resultBuffer);
	Sigmoid_Derivative_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(Sigmoid_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_ReLU_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	ReLU_Derivative_Kernel.setArg(0, tensorBuffer);
	ReLU_Derivative_Kernel.setArg(1, resultBuffer);
	ReLU_Derivative_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(ReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_LeakyReLU_Derivative(Vector<double> tensor, Vector<double> &result, double alpha)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(double), &alpha);

	// set the arguments
	LeakyReLU_Derivative_Kernel.setArg(0, tensorBuffer);
	LeakyReLU_Derivative_Kernel.setArg(1, resultBuffer);
	LeakyReLU_Derivative_Kernel.setArg(2, GetSize);
	LeakyReLU_Derivative_Kernel.setArg(3, alphaBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(LeakyReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_Tanh_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Tanh_Derivative_Kernel.setArg(0, tensorBuffer);
	Tanh_Derivative_Kernel.setArg(1, resultBuffer);
	Tanh_Derivative_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(Tanh_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_SoftMax_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Softmax_Derivative_Kernel.setArg(0, tensorBuffer);
	Softmax_Derivative_Kernel.setArg(1, resultBuffer);
	Softmax_Derivative_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(Softmax_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

static void GPU_GeLU_Derivative(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.GetSize() * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	GeLU_Derivative_Kernel.setArg(0, tensorBuffer);
	GeLU_Derivative_Kernel.setArg(1, resultBuffer);
	GeLU_Derivative_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(GeLU_Derivative_Kernel, cl::NullRange, cl::NDRange(GetSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, GetSize * sizeof(double), result.GetData());
}

// Loss functions
static void GPU_MeanSquaredError(Vector<double> output, Vector<double> target, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Sub(output, target, temp);
	GPU_Mul(temp, temp, temp);
	GPU_Sum_Fast(temp, result);
	result = result / output.GetSize();
}

static void GPU_MeanAbsoluteError(Vector<double> output, Vector<double> target, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Sub(output, target, temp);
	for (int i = 0; i < temp.GetSize(); i++)
	{
		if (temp[i] < 0)
		{
			temp[i] = -temp[i];
		}
	}
	GPU_Sum_Fast(temp, result);
	result = result / output.GetSize();
}

static void GPU_LogLoss(Vector<double> output, Vector<double> target, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Mul_Scalar(output, -1, temp);
	GPU_Mul_Scalar(temp, 1, temp);
	GPU_Mul(target, temp, temp);
	GPU_Sum_Fast(temp, result);
	result = result / output.GetSize();
}

static void GPU_CrossEntropy(Vector<double> output, Vector<double> target, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Mul_Scalar(output, -1, temp);
	GPU_Mul_Scalar(temp, 1, temp);
	GPU_Mul(target, temp, temp);
	GPU_Sum_Fast(temp, result);
	result = result / output.GetSize();
}

// Loss derivatives
static void GPU_MeanSquaredError_Derivative(Vector<double> output, Vector<double> target, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	GPU_Sub(output, target, result);
	GPU_Mul_Scalar(result, 2.0 / output.GetSize(), result);
}

static void GPU_MeanAbsoluteError_Derivative(Vector<double> output, Vector<double> target, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	GPU_Sub(output, target, result);
	for (int i = 0; i < result.GetSize(); i++)
	{
		if (result[i] < 0)
		{
			result[i] = -1;
		}
		else
		{
			result[i] = 1;
		}
	}
	GPU_Mul_Scalar(result, 1.0 / output.GetSize(), result);
}

static void GPU_LogLoss_Derivative(Vector<double> output, Vector<double> target, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Mul_Scalar(output, -1, temp);
	GPU_Mul_Scalar(temp, 1, temp);
	GPU_Div(target, temp, result);
	GPU_Mul_Scalar(result, 1.0 / output.GetSize(), result);
}

static void GPU_CrossEntropy_Derivative(Vector<double> output, Vector<double> target, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp.resize(output.GetSize());
	GPU_Mul_Scalar(output, -1, temp);
	GPU_Mul_Scalar(temp, 1, temp);
	GPU_Div(target, temp, result);
	GPU_Mul_Scalar(result, 1.0 / output.GetSize(), result);
}

static void GPU_Max_Step(Vector<double> tensor, Vector<double> &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	// create the buffers
	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.GetSize() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (tensor.GetSize() / 2) * sizeof(double));

	int GetSize = tensor.GetSize();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.GetSize() * sizeof(double), tensor.GetData());

	// set the arguments
	Max_Step_Kernel.setArg(0, tensorBuffer);
	Max_Step_Kernel.setArg(1, resultBuffer);
	Max_Step_Kernel.setArg(2, GetSize);

	// execute the kernel
	queue.enqueueNDRangeKernel(Max_Step_Kernel, cl::NullRange, cl::NDRange(GetSize / 2), cl::NullRange);

	// read the result
	result.resize(GetSize / 2);
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (tensor.GetSize() / 2) * sizeof(double), result.GetData());
}

static void GPU_Max(Vector<double> tensor, double &result)
{
	ZoneScoped;
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	Vector<double> temp;
	temp = tensor;
	while (temp.GetSize() > 1)
	{
		Vector<double> temp2;
		temp2.resize(temp.GetSize() / 2);
		GPU_Max_Step(temp, temp2);
		temp = temp2;
	}
	result = temp[0];
}
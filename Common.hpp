#pragma once
#define TRACY_ENABLE
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <filesystem>
#include <tracy/Tracy.hpp>
#include "./Kernels/Kernels.hpp"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "../include/CL/opencl.hpp"
// #include <android/log.h>

namespace Common
{

	static float RandRange(float min, float max)
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(min, max);
		return dis(gen);
	}

	// function to read the file into a string

	static std::string slurp(std::ifstream &in)
	{
		std::ostringstream sstr;
		sstr << in.rdbuf();
		return sstr.str();
	}

	namespace GPU
	{
		// OpenCL variables
		static std::vector<cl::Platform> platforms;
		static std::vector<cl::Device> devices;
		static cl::Context context;
		static cl::CommandQueue queue;
		static bool isInitialized = false;
		static bool isAvailable = false;
		static int maxWorkGroupSize;

		// function to print the build error

		static std::string getClBuildError(std::string KernelName, cl_int err, cl::Program &Program, cl::Device &Device, int line)
		{
			char *buff_erro;
			cl_int errcode;
			size_t build_log_len;
			errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
			if (errcode)
			{
				throw std::runtime_error("clGetProgramBuildInfo failed at line " + std::to_string(line) + "\n");
			}

			buff_erro = (char *)malloc(build_log_len);
			if (!buff_erro)
			{
				throw std::runtime_error("malloc failed at line " + std::to_string(line) + "\n");
			}

			errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
			if (errcode)
			{
				throw std::runtime_error("clGetProgramBuildInfo failed at line " + std::to_string(line) + "\n");
			}

			std::string error = "Build log for " + KernelName + ":\n" + buff_erro + "\n";
			free(buff_erro);
			error += "clBuildProgram failed\n";
			return error;
		}

		static const char *getErrorString(cl_int error)
		{
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

		class GPUException : public std::exception
		{
			std::string s_what;

		public:
			GPUException(cl_int err, int line, std::string kernel_name, std::string prepend)
			{
				s_what = prepend;
				s_what += "Error Loading " + kernel_name + " at line " + std::to_string(line) + ": " + getErrorString(err) + "\n";
			}
			GPUException(cl_int err, int line, std::string prepend)
			{
				s_what = prepend;
				s_what += "Error at line " + std::to_string(line) + ": " + getErrorString(err) + "\n";
			}
			GPUException(cl_int err, int line, std::string kernel_name, cl::Program &Program, cl::Device &Device, std::string prepend)
			{
				try
				{
					s_what = prepend;
					s_what += getClBuildError(kernel_name, err, Program, Device, line);
				}
				catch (std::exception &e)
				{
					s_what = e.what();
				}
			}
			GPUException(std::string message, int line)
			{
				s_what = message + " at line " + std::to_string(line);
			}

			const char *what()
			{
				return s_what.c_str();
			}
		};

		class GPUUninitializedException : public GPUException
		{
		public:
			GPUUninitializedException(int line) : GPUException("GPU not initialized", line) {}
		};

		class GPUUnavailableException : public GPUException
		{
		public:
			GPUUnavailableException(int line) : GPUException("GPU not available", line) {}
		};

		class KernelBuildException : public GPUException
		{
		public:
			KernelBuildException(std::string kernel_name, cl_int err, cl::Program &Program, cl::Device &Device, int line) : GPUException(err, line, kernel_name, Program, Device, "Could not build program: ") {}
		};

		class KernelCreationException : public GPUException
		{
		public:
			KernelCreationException(std::string kernel_name, cl_int err, int line) : GPUException(err, line, kernel_name, "Could not create kernel: ") {}
		};

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
		static cl::Kernel CalcZs_Kernel;
		static cl::Program CalcZs_Program;

		static cl::Kernel DCalcZs_dx_Kernel;
		static cl::Program DCalcZs_dx_Program;

		static cl::Kernel DCalcZs_dw_Kernel;
		static cl::Program DCalcZs_dw_Program;

		static cl::Kernel Max_Step_Kernel;
		static cl::Program Max_Step_Program;

		// flatten kernel
		static cl::Kernel Flatten2D_Kernel;
		static cl::Program Flatten2D_Program;
		static cl::Kernel Flatten3D_Kernel;
		static cl::Program Flatten3D_Program;

		// reshape kernel
		static cl::Kernel Reshape2D_Kernel;
		static cl::Program Reshape2D_Program;
		static cl::Kernel Reshape3D_Kernel;
		static cl::Program Reshape3D_Program;

		// function to initialize OpenCL and create the kernels and programs

		static void MakeKernel(
			std::string kernel_name,
			std::string kernel_code,
			cl::Kernel &kernel,
			cl::Program &program,
			int line)
		{
			cl::Program::Sources sources;
			sources.push_back({kernel_code.c_str(), kernel_code.length()});
			program = cl::Program(context, sources);
			cl_int err = program.build(devices);
			if (err != CL_SUCCESS)
			{
				throw KernelBuildException(kernel_name, err, program, devices[0], line);
			}
			kernel = cl::Kernel(program, kernel_name.c_str(), &err);
			if (err != CL_SUCCESS)
			{
				throw KernelCreationException(kernel_name, err, line);
			}
		}

		static bool IsAvailable(bool throw_e = false)
		{
			static bool firstCall = true;
			if (firstCall)
			{
				firstCall = false;
				cl::Platform::get(&platforms);
				if (platforms.empty())
				{
					isAvailable = false;
					if (throw_e)
						throw GPUException("No OpenCL platforms found!", __LINE__);
					return isAvailable;
				}

				platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
				if (devices.empty())
				{
					isAvailable = false;
					if (throw_e)
						throw GPUException("No GPU devices found!", __LINE__);
					return isAvailable;
				}
				isAvailable = true;
			}
			return false;
		}

		static bool IsInitialized()
		{
			return isInitialized;
		}

		static int GetMaxWorkGroupSize()
		{
			return maxWorkGroupSize;
		}

		static void InitializeOpenCL()
		{
			if (isInitialized)
			{
				return;
			}
			if (!IsAvailable())
			{
				throw GPUUnavailableException(__LINE__);
			}

			// get the max work group size
			maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

			char ext_str[8192];
			size_t ext_str_len;

			// setup context and queue for printing to console
			context = cl::Context(devices);

			queue = cl::CommandQueue(context, devices[0]);

			// get device extensions
			clGetDeviceInfo(devices[0](), CL_DEVICE_EXTENSIONS, 8192, ext_str, &ext_str_len);
			// printf("HeartRateDevice extensions: %s\n", ext_str);

			// vector kernels
			MakeKernel("Sum_Step", Sum_Step_Kernel_Code, Sum_Step_Kernel, Sum_Step_Program, __LINE__);

			MakeKernel("Mul", Mul_Kernel_Code, Mul_Kernel, Mul_Program, __LINE__);

			MakeKernel("Mul_Scalar", Mul_Scalar_Kernel_Code, Mul_Scalar_Kernel, Mul_Scalar_Program, __LINE__);

			MakeKernel("Sub", Sub_Kernel_Code, Sub_Kernel, Sub_Program, __LINE__);

			MakeKernel("Add", Add_Kernel_Code, Add_Kernel, Add_Program, __LINE__);

			MakeKernel("Div", Div_Kernel_Code, Div_Kernel, Div_Program, __LINE__);

			MakeKernel("Abs", Abs_Kernel_Code, Abs_Kernel, Abs_Program, __LINE__);

			MakeKernel("Abs_Derivative", Abs_Derivative_Kernel_Code, Abs_Derivative_Kernel, Abs_Derivative_Program, __LINE__);

			// matrix kernels
			MakeKernel("Matrix_Transpose", Matrix_Transpose_Kernel_Code, Matrix_Transpose_Kernel, Matrix_Transpose_Program, __LINE__);

			// tensor kernels
			// MakeKernel("Tensor_Block", "./MLLib/Kernels/Math/tensor/Tensor_Block.cl", Tensor_Block_Kernel, Tensor_Block_Program,__LINE__);

			// MakeKernel("Tensor_Transpose", "./MLLib/Kernels/Math/tensor/Tensor_Transpose.cl", Tensor_Transpose_Kernel, Tensor_Transpose_Program,__LINE__);

			// MakeKernel("Tensor_Dot", "./MLLib/Kernels/Math/tensor/Tensor_Dot.cl", Tensor_Dot_Kernel, Tensor_Dot_Program,__LINE__);

			// MakeKernel("Tensor_Conv", "./MLLib/Kernels/Math/tensor/Tensor_Conv.cl", Tensor_Conv_Kernel, Tensor_Conv_Program,__LINE__);

			// activation functions
			MakeKernel("Sigmoid", Sigmoid_Kernel_Code, Sigmoid_Kernel, Sigmoid_Program, __LINE__);

			MakeKernel("ReLU", ReLU_Kernel_Code, ReLU_Kernel, ReLU_Program, __LINE__);

			MakeKernel("LeakyReLU", LeakyReLU_Kernel_Code, LeakyReLU_Kernel, LeakyReLU_Program, __LINE__);

			MakeKernel("Tanh", Tanh_Kernel_Code, Tanh_Kernel, Tanh_Program, __LINE__);

			MakeKernel("SoftMax", SoftMax_Kernel_Code, Softmax_Kernel, Softmax_Program, __LINE__);

			MakeKernel("GeLU", GeLU_Kernel_Code, GeLU_Kernel, GeLU_Program, __LINE__);

			// activation derivatives
			MakeKernel("Sigmoid_Derivative", Sigmoid_Derivative_Kernel_Code, Sigmoid_Derivative_Kernel, Sigmoid_Derivative_Program, __LINE__);

			MakeKernel("ReLU_Derivative", ReLU_Derivative_Kernel_Code, ReLU_Derivative_Kernel, ReLU_Derivative_Program, __LINE__);

			MakeKernel("LeakyReLU_Derivative", LeakyReLU_Derivative_Kernel_Code, LeakyReLU_Derivative_Kernel, LeakyReLU_Derivative_Program, __LINE__);

			MakeKernel("Tanh_Derivative", Tanh_Derivative_Kernel_Code, Tanh_Derivative_Kernel, Tanh_Derivative_Program, __LINE__);

			MakeKernel("SoftMax_Derivative", SoftMax_Derivative_Kernel_Code, Softmax_Derivative_Kernel, Softmax_Derivative_Program, __LINE__);

			MakeKernel("GeLU_Derivative", GeLU_Derivative_Kernel_Code, GeLU_Derivative_Kernel, GeLU_Derivative_Program, __LINE__);

			// Util
			MakeKernel("CalcZs", CalcZs_Kernel_Code, CalcZs_Kernel, CalcZs_Program, __LINE__);
			MakeKernel("DCalcZs_dx", DCalcZs_dx_Kernel_Code, DCalcZs_dx_Kernel, DCalcZs_dx_Program, __LINE__);
			MakeKernel("DCalcZs_dw", DCalcZs_dw_Kernel_Code, DCalcZs_dw_Kernel, DCalcZs_dw_Program, __LINE__);
			MakeKernel("Max_Step", Max_Step_Kernel_Code, Max_Step_Kernel, Max_Step_Program, __LINE__);

			// flatten kernels
			MakeKernel("Flatten2D", Flatten2D_Kernel_Code, Flatten2D_Kernel, Flatten2D_Program, __LINE__);
			MakeKernel("Flatten3D", Flatten3D_Kernel_Code, Flatten3D_Kernel, Flatten3D_Program, __LINE__);

			// reshape kernels
			MakeKernel("Reshape2D", Reshape2D_Kernel_Code, Reshape2D_Kernel, Reshape2D_Program, __LINE__);
			MakeKernel("Reshape3D", Reshape3D_Kernel_Code, Reshape3D_Kernel, Reshape3D_Program, __LINE__);

			// set the flag
			isInitialized = true;
		}

		static void Flatten(std::vector<std::vector<float>> data, std::vector<float> &result)
		{
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			int size = data.size() * data[0].size();

			result.resize(size);

			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

			cl_int err;
			err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(float), data.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten2D_Kernel", "Could not write to dataBuffer");
			}

			err = Flatten2D_Kernel.setArg(0, dataBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten2D_Kernel", "Could not set dataBuffer");
			}
			err = Flatten2D_Kernel.setArg(1, resultBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten2D_Kernel", "Could not set resultBuffer");
			}
			err = Flatten2D_Kernel.setArg(2, size);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten2D_Kernel", "Could not set size");
			}

			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			err = queue.enqueueNDRangeKernel(Flatten2D_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
			if (err != CL_SUCCESS)
			{
				throw std::runtime_error("Could not execute the kernel");
				throw GPUException(err, __LINE__, "Flatten2D_Kernel", "Could not execute the kernel");
			}

			queue.finish();
		}

		static void Flatten(std::vector<std::vector<std::vector<float>>> &data, std::vector<float> &result)
		{
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			int size = data.size() * data[0].size() * data[0][0].size();

			result.resize(size);

			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

			cl_int err;
			err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(float), data.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten3D_Kernel", "Could not write to dataBuffer");
			}

			err = Flatten3D_Kernel.setArg(0, dataBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten3D_Kernel", "Could not set dataBuffer");
			}
			err = Flatten3D_Kernel.setArg(1, resultBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten3D_Kernel", "Could not set resultBuffer");
			}
			err = Flatten3D_Kernel.setArg(2, size);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten3D_Kernel", "Could not set size");
			}

			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			err = queue.enqueueNDRangeKernel(Flatten3D_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Flatten3D_Kernel", "Could not execute the kernel");
			}

			queue.finish();
		}

		static void Reshape(std::vector<float> data, std::vector<std::vector<float>> &result, int rows, int cols)
		{
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			int size = data.size();

			result.resize(rows);
			for (int i = 0; i < rows; i++)
			{
				result[i].resize(cols);
			}

			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

			cl_int err;
			err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(float), data.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not write to dataBuffer");
			}

			err = Reshape2D_Kernel.setArg(0, dataBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not set dataBuffer");
			}
			err = Reshape2D_Kernel.setArg(1, resultBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not set resultBuffer");
			}
			err = Reshape2D_Kernel.setArg(2, size);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not set size");
			}
			err = Reshape2D_Kernel.setArg(3, rows);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not set rows");
			}
			err = Reshape2D_Kernel.setArg(4, cols);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not set cols");
			}

			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			err = queue.enqueueNDRangeKernel(Reshape2D_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape2D_Kernel", "Could not execute the kernel");
			}

			queue.finish();
		}

		static void Reshape(std::vector<float> data, std::vector<std::vector<std::vector<float>>> &result, int rows, int cols, int depth)
		{
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			int size = data.size();

			result.resize(rows);
			for (int i = 0; i < rows; i++)
			{
				result[i].resize(cols);
				for (int j = 0; j < cols; j++)
				{
					result[i][j].resize(depth);
				}
			}

			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

			cl_int err;
			err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(float), data.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not write to dataBuffer");
			}

			err = Reshape3D_Kernel.setArg(0, dataBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set dataBuffer");
			}
			err = Reshape3D_Kernel.setArg(1, resultBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set resultBuffer");
			}
			err = Reshape3D_Kernel.setArg(2, size);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set size");
			}
			err = Reshape3D_Kernel.setArg(3, rows);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set rows");
			}
			err = Reshape3D_Kernel.setArg(4, cols);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set cols");
			}
			err = Reshape3D_Kernel.setArg(5, depth);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not set depth");
			}

			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			err = queue.enqueueNDRangeKernel(Reshape3D_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Reshape3D_Kernel", "Could not execute the kernel");
			}

			queue.finish();
		}

		// vector functions
		static void Sum_Step(std::vector<float> data, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			result.resize(data.size() / 2);

			// create the buffers
			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, data.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (data.size() / 2) * sizeof(float));

			int size = data.size();

			// write the data to the buffers
			cl_int err;
			err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.size() * sizeof(float), data.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not write to dataBuffer");
			}

			// set the arguments
			err = Sum_Step_Kernel.setArg(0, dataBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set dataBuffer");
			}
			err = Sum_Step_Kernel.setArg(1, resultBuffer);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set resultBuffer");
			}
			err = Sum_Step_Kernel.setArg(2, size);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set size");
			}

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > (size / 2))
			{
				workGroupSize = (size / 2);
			}

			// execute the kernel
			err = queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not execute the kernel");
			}

			// resize the result vector
			result.resize(size / 2);

			// wait for the kernel to finish executing
			queue.finish();

			// read the result
			err = queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (data.size() / 2) * sizeof(float), result.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not read from resultBuffer");
			}

			// if the result is not a power of 2, add the last element
			if (data.size() % 2 != 0)
			{
				int size = data.size();
				int index{size - 1};
				result.push_back(data[index]);
			}
		}

		static void Sum(std::vector<float> data, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp = data;
			while (temp.size() > 1)
			{
				std::vector<float> temp2;
				temp2.resize(temp.size() / 2);
				Sum_Step(temp, temp2);
				temp = temp2;
			}
			result = temp[0];
		}

		static void Sum_Fast(std::vector<float> data, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			// create the buffers
			cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, data.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (data.size() / 2) * sizeof(float));

			int size = data.size();
			int times = data.size() / 2;

			for (int i = 0; i < times; i++)
			{
				// write the data to the buffers
				cl_int err;
				err = queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.size() * sizeof(float), data.data());
				if (err != CL_SUCCESS)
				{
					throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not write to dataBuffer");
				}

				// set the arguments
				err = Sum_Step_Kernel.setArg(0, dataBuffer);
				if (err != CL_SUCCESS)
				{
					throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set dataBuffer");
				}
				err = Sum_Step_Kernel.setArg(1, resultBuffer);
				if (err != CL_SUCCESS)
				{
					throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set resultBuffer");
				}
				err = Sum_Step_Kernel.setArg(2, size);
				if (err != CL_SUCCESS)
				{
					throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not set size");
				}

				// get the max work group size
				size_t workGroupSize = GetMaxWorkGroupSize();
				if (workGroupSize > (size / 2))
				{
					workGroupSize = (size / 2);
				}

				// execute the kernel
				err = queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);
				if (err != CL_SUCCESS)
				{
					throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not execute the kernel");
				}

				size = size / 2;
			}

			// wait for the kernel to finish executing
			queue.finish();

			std::vector<float> temp;
			temp.resize(data.size() / 2);

			// read the result
			cl_int err = queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (data.size() / 2) * sizeof(float), temp.data());
			if (err != CL_SUCCESS)
			{
				throw GPUException(err, __LINE__, "Sum_Step_Kernel", "Could not read from resultBuffer");
			}

			result = temp[0];

			// if the result is not a power of 2, add the last element
			if (data.size() % 2 != 0)
			{
				result += data[data.size() - 1];
			}
		}

		static void Mul(std::vector<float> data1, std::vector<float> data2, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			// check if the size of the data1 is equal to the size of the data2
			if (data1.size() != data2.size())
			{
				throw std::invalid_argument("The size of the data1 is not equal to the size of the data2");
			}

			int size = data1.size();

			// create the buffers
			cl::Buffer data1Buffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer data2Buffer(context, CL_MEM_READ_ONLY, size * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

			// write the data to the buffers
			queue.enqueueWriteBuffer(data1Buffer, CL_TRUE, 0, size * sizeof(float), data1.data());
			queue.enqueueWriteBuffer(data2Buffer, CL_TRUE, 0, size * sizeof(float), data2.data());

			// set the arguments
			Mul_Kernel.setArg(0, data1Buffer);
			Mul_Kernel.setArg(1, data2Buffer);
			Mul_Kernel.setArg(2, resultBuffer);
			Mul_Kernel.setArg(3, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Mul_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// resize the result vector
			result.resize(size);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void MulSum(std::vector<float> tensor1, std::vector<float> tensor2, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(tensor1.size());
			Mul(tensor1, tensor2, temp);
			Sum_Fast(temp, result);
		}

		static void Mul_Scalar(std::vector<float> tensor, float scalar, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Mul_Scalar_Kernel.setArg(0, tensorBuffer);
			Mul_Scalar_Kernel.setArg(1, scalar);
			Mul_Scalar_Kernel.setArg(2, resultBuffer);
			Mul_Scalar_Kernel.setArg(3, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Mul_Scalar_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// resize the result vector
			result.resize(size);

			queue.finish();

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Sub(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(float));
			cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(float));

			int size = tensor1.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(float), tensor1.data());
			queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(float), tensor2.data());

			// set the arguments
			Sub_Kernel.setArg(0, tensor1Buffer);
			Sub_Kernel.setArg(1, tensor2Buffer);
			Sub_Kernel.setArg(2, resultBuffer);
			Sub_Kernel.setArg(3, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Sub_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// resize the result vector
			result.resize(size);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Add(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(float));
			cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(float));

			int size = tensor1.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(float), tensor1.data());
			queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(float), tensor2.data());

			// set the arguments
			Add_Kernel.setArg(0, tensor1Buffer);
			Add_Kernel.setArg(1, tensor2Buffer);
			Add_Kernel.setArg(2, resultBuffer);
			Add_Kernel.setArg(3, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Add_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// resize the result vector
			result.resize(size);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Div(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(float));
			cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(float));

			int size = tensor1.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(float), tensor1.data());
			queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(float), tensor2.data());

			// set the arguments
			Div_Kernel.setArg(0, tensor1Buffer);
			Div_Kernel.setArg(1, tensor2Buffer);
			Div_Kernel.setArg(2, resultBuffer);
			Div_Kernel.setArg(3, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Div_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// resize the result vector
			result.resize(size);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Abs(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Abs_Kernel.setArg(0, tensorBuffer);
			Abs_Kernel.setArg(1, resultBuffer);
			Abs_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Abs_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Abs_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Abs_Derivative_Kernel.setArg(0, tensorBuffer);
			Abs_Derivative_Kernel.setArg(1, resultBuffer);
			Abs_Derivative_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Abs_Derivative_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		// matrix functions
		static void TransposeMatrix(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>> &Result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			int Rows = matrix.size();
			int Cols = matrix[0].size();

			std::vector<float> _Result(Rows * Cols);
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					_Result[j * Rows + i] = matrix[i][j];
				}
			}

			// create the buffers
			cl::Buffer matrixBuffer(context, CL_MEM_READ_ONLY, Rows * Cols * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Rows * Cols * sizeof(float));

			// write the data to the buffers
			queue.enqueueWriteBuffer(matrixBuffer, CL_TRUE, 0, Rows * Cols * sizeof(float), _Result.data());

			// set the arguments
			Matrix_Transpose_Kernel.setArg(0, matrixBuffer);
			Matrix_Transpose_Kernel.setArg(1, resultBuffer);
			Matrix_Transpose_Kernel.setArg(2, Rows);
			Matrix_Transpose_Kernel.setArg(3, Cols);

			// execute the kernel
			queue.enqueueNDRangeKernel(Matrix_Transpose_Kernel, cl::NullRange, cl::NDRange(Rows, Cols), cl::NullRange);

			queue.finish();

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Rows * Cols * sizeof(float), _Result.data());

			Result.resize(Cols);
			for (int i = 0; i < Cols; i++)
			{
				Result[i].resize(Rows);
				for (int j = 0; j < Rows; j++)
				{
					Result[i][j] = _Result[i * Rows + j];
				}
			}
		}

		// Activation functions
		static void Sigmoid(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Sigmoid_Kernel.setArg(0, tensorBuffer);
			Sigmoid_Kernel.setArg(1, resultBuffer);
			Sigmoid_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Sigmoid_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void ReLU(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			ReLU_Kernel.setArg(0, tensorBuffer);
			ReLU_Kernel.setArg(1, resultBuffer);
			ReLU_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(ReLU_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void LeakyReLU(std::vector<float> tensor, std::vector<float> &result, float alpha)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(float), &alpha);

			// set the arguments
			LeakyReLU_Kernel.setArg(0, tensorBuffer);
			LeakyReLU_Kernel.setArg(1, resultBuffer);
			LeakyReLU_Kernel.setArg(2, size);
			LeakyReLU_Kernel.setArg(3, alphaBuffer);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(LeakyReLU_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Tanh(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Tanh_Kernel.setArg(0, tensorBuffer);
			Tanh_Kernel.setArg(1, resultBuffer);
			Tanh_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Tanh_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void SoftMax(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Softmax_Kernel.setArg(0, tensorBuffer);
			Softmax_Kernel.setArg(1, resultBuffer);
			Softmax_Kernel.setArg(2, size);

			float max_val = tensor[0];
			for (int i = 1; i < tensor.size(); ++i)
			{
				if (tensor[i] > max_val)
				{
					max_val = tensor[i];
				}
			}

			float sum = 0.0f;
			for (int i = 0; i < tensor.size(); ++i)
			{
				sum += exp(tensor[i] - max_val);
			}

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(Softmax_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void GeLU(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			GeLU_Kernel.setArg(0, tensorBuffer);
			GeLU_Kernel.setArg(1, resultBuffer);
			GeLU_Kernel.setArg(2, size);

			// get the max work group size
			size_t workGroupSize = GetMaxWorkGroupSize();
			if (workGroupSize > size)
			{
				workGroupSize = size;
			}

			// execute the kernel
			queue.enqueueNDRangeKernel(GeLU_Kernel, cl::NullRange, cl::NDRange(workGroupSize), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		// Activation derivatives
		static void Sigmoid_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Sigmoid_Derivative_Kernel.setArg(0, tensorBuffer);
			Sigmoid_Derivative_Kernel.setArg(1, resultBuffer);
			Sigmoid_Derivative_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(Sigmoid_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void ReLU_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			ReLU_Derivative_Kernel.setArg(0, tensorBuffer);
			ReLU_Derivative_Kernel.setArg(1, resultBuffer);
			ReLU_Derivative_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(ReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void LeakyReLU_Derivative(std::vector<float> tensor, std::vector<float> &result, float alpha)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(float), &alpha);

			// set the arguments
			LeakyReLU_Derivative_Kernel.setArg(0, tensorBuffer);
			LeakyReLU_Derivative_Kernel.setArg(1, resultBuffer);
			LeakyReLU_Derivative_Kernel.setArg(2, size);
			LeakyReLU_Derivative_Kernel.setArg(3, alphaBuffer);

			// execute the kernel
			queue.enqueueNDRangeKernel(LeakyReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void Tanh_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Tanh_Derivative_Kernel.setArg(0, tensorBuffer);
			Tanh_Derivative_Kernel.setArg(1, resultBuffer);
			Tanh_Derivative_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(Tanh_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void SoftMax_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Softmax_Derivative_Kernel.setArg(0, tensorBuffer);
			Softmax_Derivative_Kernel.setArg(1, resultBuffer);
			Softmax_Derivative_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(Softmax_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		static void GeLU_Derivative(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			GeLU_Derivative_Kernel.setArg(0, tensorBuffer);
			GeLU_Derivative_Kernel.setArg(1, resultBuffer);
			GeLU_Derivative_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(GeLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

			// read the result
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(float), result.data());
		}

		// Loss functions
		static void MeanSquaredError(std::vector<float> output, std::vector<float> target, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Sub(output, target, temp);
			Mul(temp, temp, temp);
			Sum_Fast(temp, result);
			result = result / output.size();
		}

		static void MeanAbsoluteError(std::vector<float> output, std::vector<float> target, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Sub(output, target, temp);
			for (int i = 0; i < temp.size(); i++)
			{
				if (temp[i] < 0)
				{
					temp[i] = -temp[i];
				}
			}
			Sum_Fast(temp, result);
			result = result / output.size();
		}

		static void LogLoss(std::vector<float> output, std::vector<float> target, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Mul_Scalar(output, -1, temp);
			Mul_Scalar(temp, 1, temp);
			Mul(target, temp, temp);
			Sum_Fast(temp, result);
			result = result / output.size();
		}

		static void CrossEntropy(std::vector<float> output, std::vector<float> target, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Mul_Scalar(output, -1, temp);
			Mul_Scalar(temp, 1, temp);
			Mul(target, temp, temp);
			Sum_Fast(temp, result);
			result = result / output.size();
		}

		// Loss derivatives
		static void MeanSquaredError_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			Sub(output, target, result);
			Mul_Scalar(result, 2.0 / output.size(), result);
		}

		static void MeanAbsoluteError_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			Sub(output, target, result);
			for (int i = 0; i < result.size(); i++)
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
			Mul_Scalar(result, 1.0 / output.size(), result);
		}

		static void LogLoss_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Mul_Scalar(output, -1, temp);
			Mul_Scalar(temp, 1, temp);
			Div(target, temp, result);
			Mul_Scalar(result, 1.0 / output.size(), result);
		}

		static void CrossEntropy_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp.resize(output.size());
			Mul_Scalar(output, -1, temp);
			Mul_Scalar(temp, 1, temp);
			Div(target, temp, result);
			Mul_Scalar(result, 1.0 / output.size(), result);
		}

		static void CalcZs(std::vector<float> x, std::vector<std::vector<float>> w, std::vector<float> b, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> w_flat(w.size() * w[0].size());
			Flatten(w, w_flat);
			int WSizeX = w.size();
			int WSizeY = w[0].size();
			int XSize = x.size();
			int BSize = b.size();
			int ZSize = BSize;

			cl::Buffer xBuffer(context, CL_MEM_READ_ONLY, x.size() * sizeof(float));
			cl::Buffer wBuffer(context, CL_MEM_READ_ONLY, w_flat.size() * sizeof(float));
			cl::Buffer bBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, ZSize * sizeof(float));

			// write the data to the buffers
			queue.enqueueWriteBuffer(xBuffer, CL_TRUE, 0, x.size() * sizeof(float), x.data());
			queue.enqueueWriteBuffer(wBuffer, CL_TRUE, 0, w_flat.size() * sizeof(float), w_flat.data());
			queue.enqueueWriteBuffer(bBuffer, CL_TRUE, 0, b.size() * sizeof(float), b.data());

			// set the arguments
			CalcZs_Kernel.setArg(0, xBuffer);
			CalcZs_Kernel.setArg(1, XSize);
			CalcZs_Kernel.setArg(2, wBuffer);
			CalcZs_Kernel.setArg(3, WSizeX);
			CalcZs_Kernel.setArg(4, WSizeY);
			CalcZs_Kernel.setArg(5, bBuffer);
			CalcZs_Kernel.setArg(6, BSize);
			CalcZs_Kernel.setArg(7, resultBuffer);
			CalcZs_Kernel.setArg(8, ZSize);

			queue.enqueueNDRangeKernel(CalcZs_Kernel, cl::NullRange, cl::NDRange(ZSize), cl::NullRange);

			// read the result
			result.resize(ZSize);
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, ZSize * sizeof(float), result.data());
		}

		static void DCalcZs_dx(std::vector<std::vector<float>> w, std::vector<float> &result)
		{
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}
			std::vector<float> w_flat(w.size() * w[0].size());
			Flatten(w, w_flat);
			int WSizeX = w.size();
			int WSizeY = w[0].size();
			int XSize = w[0].size();

			cl::Buffer wBuffer(context, CL_MEM_READ_ONLY, w_flat.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, XSize * sizeof(float));

			// write the data to the buffers
			queue.enqueueWriteBuffer(wBuffer, CL_TRUE, 0, w_flat.size() * sizeof(float), w_flat.data());

			// set the arguments
			DCalcZs_dx_Kernel.setArg(0, wBuffer);
			DCalcZs_dx_Kernel.setArg(1, WSizeX);
			DCalcZs_dx_Kernel.setArg(2, WSizeY);
			DCalcZs_dx_Kernel.setArg(3, resultBuffer);
			DCalcZs_dx_Kernel.setArg(4, XSize);

			queue.enqueueNDRangeKernel(DCalcZs_dx_Kernel, cl::NullRange, cl::NDRange(XSize), cl::NullRange);

			// read the result
			result.resize(XSize);
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, XSize * sizeof(float), result.data());
		}

		static void DCalcZs_dw(std::vector<float> x, int outputs, std::vector<std::vector<float>> &result)
		{
		}

		static void Max_Step(std::vector<float> tensor, std::vector<float> &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			// create the buffers
			cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(float));
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (tensor.size() / 2) * sizeof(float));

			int size = tensor.size();

			// write the data to the buffers
			queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(float), tensor.data());

			// set the arguments
			Max_Step_Kernel.setArg(0, tensorBuffer);
			Max_Step_Kernel.setArg(1, resultBuffer);
			Max_Step_Kernel.setArg(2, size);

			// execute the kernel
			queue.enqueueNDRangeKernel(Max_Step_Kernel, cl::NullRange, cl::NDRange(size / 2), cl::NullRange);

			// read the result
			result.resize(size / 2);
			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (tensor.size() / 2) * sizeof(float), result.data());
		}

		static void Max(std::vector<float> tensor, float &result)
		{
			// check if OpenCL is initialized
			if (IsAvailable() == false)
			{
				throw GPUUnavailableException(__LINE__);
			}
			if (IsInitialized() == false)
			{
				throw GPUUninitializedException(__LINE__);
			}

			std::vector<float> temp;
			temp = tensor;
			while (temp.size() > 1)
			{
				std::vector<float> temp2;
				temp2.resize(temp.size() / 2);
				Max_Step(temp, temp2);
				temp = temp2;
			}

			result = temp[0];
		}
	};

	class BestPerformanceException : public std::exception
	{
		int line;

	public:
		BestPerformanceException(int line)
		{
			this->line = line;
		}

		const char *what() const throw()
		{
			return ("GPU is available but not initialized. Please call GPU::Initialize() before calling this function for best performance. Line: " + std::to_string(line)).c_str();
		}
	};

	static void Flatten(std::vector<std::vector<float>> data, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Flatten(data, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data[i].size(); j++)
			{
				result.push_back(data[i][j]);
			}
		}
	}

	static void Flatten(std::vector<std::vector<std::vector<float>>> &data, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Flatten(data, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data[i].size(); j++)
			{
				for (int k = 0; k < data[i][j].size(); k++)
				{
					result.push_back(data[i][j][k]);
				}
			}
		}
	}

	static void Reshape(std::vector<float> data, std::vector<std::vector<float>> &result, int rows, int cols)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Reshape(data, result, rows, cols);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < rows; i++)
		{
			std::vector<float> temp;
			for (int j = 0; j < cols; j++)
			{
				temp.push_back(data[i * cols + j]);
			}
			result.push_back(temp);
		}
	}

	static void Reshape(std::vector<float> data, std::vector<std::vector<std::vector<float>>> &result, int rows, int cols, int depth)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Reshape(data, result, rows, cols, depth);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < rows; i++)
		{
			std::vector<std::vector<float>> temp;
			for (int j = 0; j < cols; j++)
			{
				std::vector<float> temp2;
				for (int k = 0; k < depth; k++)
				{
					temp2.push_back(data[i * cols * depth + j * depth + k]);
				}
				temp.push_back(temp2);
			}
			result.push_back(temp);
		}
	}

	static void Split(std::vector<float> data, std::vector<std::vector<float>> &result, int span)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			// GPU::Split(data, result, span);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		int size = data.size();
		int num = size / span;
		for (int i = 0; i < num; i++)
		{
			std::vector<float> temp;
			for (int j = 0; j < span; j++)
			{
				temp.push_back(data[i * span + j]);
			}
			result.push_back(temp);
		}
	}

	static float Clamp(float val, float min, float max)
	{
		if (val < min)
		{
			val = min;
		}
		else if (val > max)
		{
			val = max;
		}
		return val;
	}

	static std::vector<float> Clamp(std::vector<float> vec, float min, float max)
	{
		for (int i = 0; i < vec.size(); i++)
		{
			if (vec[i] < min)
			{
				vec[i] = min;
			}
			else if (vec[i] > max)
			{
				vec[i] = max;
			}
		}
		return vec;
	}

	static std::vector<std::vector<float>> Clamp(std::vector<std::vector<float>> vec, float min, float max)
	{
		for (int i = 0; i < vec.size(); i++)
		{
			vec[i]=Clamp(vec[i], min, max);
		}
		return vec;
	}

	static float Map(float val, float in_min, float in_max, float out_min, float out_max)
	{
		return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	static void Sum(std::vector<float> data, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Sum(data, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result = 0;
		for (int i = 0; i < data.size(); i++)
		{
			result += data[i];
		}
	}

	static void Mul(std::vector<float> data1, std::vector<float> data2, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Mul(data1, data2, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < data1.size(); i++)
		{
			result.push_back(data1[i] * data2[i]);
		}
	}

	static void MulSum(std::vector<float> tensor1, std::vector<float> tensor2, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::MulSum(tensor1, tensor2, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result = 0;
		for (int i = 0; i < tensor1.size(); i++)
		{
			result += tensor1[i] * tensor2[i];
		}
	}

	static void Mul_Scalar(std::vector<float> tensor, float scalar, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Mul_Scalar(tensor, scalar, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] * scalar);
		}
	}

	static void Sub(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Sub(tensor1, tensor2, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor1.size(); i++)
		{
			result.push_back(tensor1[i] - tensor2[i]);
		}
	}

	static void Sub(std::vector<std::vector<float>> tensor1, std::vector<std::vector<float>> tensor2, std::vector<std::vector<float>> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			for (int i = 0; i < tensor1.size(); i++)
			{
				GPU::Sub(tensor1[i], tensor2[i], result[i]);
			}
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		for (int i = 0; i < tensor1.size(); i++)
		{
			for (int j = 0; j < tensor1[i].size(); j++)
			{
				result[i][j]=tensor1[i][j] - tensor2[i][j];
			}
		}
	}

	static void Add(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Add(tensor1, tensor2, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor1.size(); i++)
		{
			result.push_back(tensor1[i] + tensor2[i]);
		}
	}

	static void Div(std::vector<float> tensor1, std::vector<float> tensor2, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Div(tensor1, tensor2, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor1.size(); i++)
		{
			result.push_back(tensor1[i] / tensor2[i]);
		}
	}

	static void Abs(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Abs(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(abs(tensor[i]));
		}
	}

	static void Abs_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Abs_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			if (tensor[i] > 0)
			{
				result.push_back(1);
			}
			else if (tensor[i] < 0)
			{
				result.push_back(-1);
			}
			else
			{
				result.push_back(0);
			}
		}
	}

	// matrix functions
	static void TransposeMatrix(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>> &Result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::TransposeMatrix(matrix, Result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		Result.clear();
		for (int i = 0; i < matrix[0].size(); i++)
		{
			std::vector<float> temp;
			for (int j = 0; j < matrix.size(); j++)
			{
				temp.push_back(matrix[j][i]);
			}
			Result.push_back(temp);
		}
	}

	// Activation functions
	static void Sigmoid(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Sigmoid(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(1.0 / (1.0 + exp(-tensor[i])));
		}
	}

	static void ReLU(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::ReLU(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] > 0 ? tensor[i] : 0);
		}
	}

	static void LeakyReLU(std::vector<float> tensor, std::vector<float> &result, float alpha)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::LeakyReLU(tensor, result, alpha);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] > 0 ? tensor[i] : alpha * tensor[i]);
		}
	}

	static void Tanh(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Tanh(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tanh(tensor[i]));
		}
	}

	static void SoftMax(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::SoftMax(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		float max_val = tensor[0];
		for (int i = 1; i < tensor.size(); ++i)
		{
			if (tensor[i] > max_val)
			{
				max_val = tensor[i];
			}
		}

		float sum = 0.0f;
		for (int i = 0; i < tensor.size(); ++i)
		{
			sum += exp(tensor[i] - max_val);
		}

		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(exp(tensor[i] - max_val) / sum);
		}
	}

	static void GeLU(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::GeLU(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(0.5 * tensor[i] * (1 + erf(tensor[i] / sqrt(2))));
		}
	}

	// Activation derivatives
	static void Sigmoid_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Sigmoid_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(exp(-tensor[i]) / ((1 + exp(-tensor[i])) * (1 + exp(-tensor[i]))));
		}
	}

	static void ReLU_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::ReLU_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] > 0 ? 1 : 0);
		}
	}

	static void LeakyReLU_Derivative(std::vector<float> tensor, std::vector<float> &result, float alpha)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::LeakyReLU_Derivative(tensor, result, alpha);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] > 0 ? 1 : alpha);
		}
	}

	static void Tanh_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Tanh_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(1 - tensor[i] * tensor[i]);
		}
	}

	static void SoftMax_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::SoftMax_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		float sum = 0.0f;
		for (int i = 0; i < tensor.size(); i++)
		{
			sum += exp(tensor[i]);
		}
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(tensor[i] * (1 - tensor[i]) / sum);
		}
	}

	static void GeLU_Derivative(std::vector<float> tensor, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::GeLU_Derivative(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < tensor.size(); i++)
		{
			result.push_back(0.5 * (1 + erf(tensor[i] / sqrt(2)) + tensor[i] * exp(-tensor[i] * tensor[i] / 2) / sqrt(2 * M_PI)));
		}
	}

	// Loss functions
	static void MeanSquaredError(std::vector<float> output, std::vector<float> target, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::MeanSquaredError(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Sub(output, target, temp);
		Mul(temp, temp, temp);
		Sum(temp, result);
		result = result / output.size();
	}

	static void MeanAbsoluteError(std::vector<float> output, std::vector<float> target, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::MeanAbsoluteError(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Sub(output, target, temp);
		for (int i = 0; i < temp.size(); i++)
		{
			if (temp[i] < 0)
			{
				temp[i] = -temp[i];
			}
		}
		Sum(temp, result);
		result = result / output.size();
	}

	static void LogLoss(std::vector<float> output, std::vector<float> target, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::LogLoss(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Mul_Scalar(output, -1, temp);
		Mul_Scalar(temp, 1, temp);
		Mul(target, temp, temp);
		Sum(temp, result);
		result = result / output.size();
	}

	static void CrossEntropy(std::vector<float> output, std::vector<float> target, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::CrossEntropy(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Mul_Scalar(output, -1, temp);
		Mul_Scalar(temp, 1, temp);
		Mul(target, temp, temp);
		Sum(temp, result);
		result = result / output.size();
	}

	// Loss derivatives
	static void MeanSquaredError_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::MeanSquaredError_Derivative(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		Sub(output, target, result);
		Mul_Scalar(result, 2.0 / output.size(), result);
	}

	static void MeanAbsoluteError_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::MeanAbsoluteError_Derivative(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		Sub(output, target, result);
		for (int i = 0; i < result.size(); i++)
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
		Mul_Scalar(result, 1.0 / output.size(), result);
	}

	static void LogLoss_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::LogLoss_Derivative(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Mul_Scalar(output, -1, temp);
		Mul_Scalar(temp, 1, temp);
		Div(target, temp, result);
		Mul_Scalar(result, 1.0 / output.size(), result);
	}

	static void CrossEntropy_Derivative(std::vector<float> output, std::vector<float> target, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::CrossEntropy_Derivative(output, target, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		std::vector<float> temp;
		temp.resize(output.size());
		Mul_Scalar(output, -1, temp);
		Mul_Scalar(temp, 1, temp);
		Div(target, temp, result);
		Mul_Scalar(result, 1.0 / output.size(), result);
	}

	static void CalcZs(std::vector<float> x, std::vector<std::vector<float>> w, std::vector<float> b, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::CalcZs(x, w, b, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result.clear();
		for (int i = 0; i < w.size(); i++)
		{
			float temp = 0;
			for (int j = 0; j < w[i].size(); j++)
			{
				temp += w[i][j] * x[j];
			}
			temp += b[i];
			result.push_back(temp);
		}
	}

	static void DCalcZs_dx(std::vector<std::vector<float>> w, std::vector<float> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::DCalcZs_dx(w, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result = std::vector<float>(w[0].size(), 0);
		for (int i = 0; i < w[0].size(); i++)
		{
			for (int j = 0; j < w.size(); j++)
			{
				result[i] += w[j][i];
			}
		}
	}

	static void DCalcZs_dw(std::vector<float> x, int outputs, std::vector<std::vector<float>> &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::DCalcZs_dw(x, outputs, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		result = std::vector<std::vector<float>>(outputs, std::vector<float>(x.size(), 0));
		for (int j = 0; j < outputs; j++)
		{
			for (int i = 0; i < x.size(); i++)
			{
				result[j][i] = x[i];
			}
		}
	}

	static void DCalcZs_db(int b_size, std::vector<float> &result)
	{
		result = std::vector<float>(b_size, 1);
	}

	static void Max(std::vector<float> tensor, float &result)
	{
		if (GPU::IsAvailable() == true && GPU::IsInitialized() == true)
		{
			GPU::Max(tensor, result);
		}
		else if (GPU::IsAvailable() == true && GPU::IsInitialized() == false)
		{
			GPU::InitializeOpenCL();
		}
		float temp;
		for (int i = 0; i < tensor.size(); i++)
		{
			if (i == 0)
			{
				temp = tensor[i];
			}
			else
			{
				if (tensor[i] > temp)
				{
					temp = tensor[i];
				}
			}
		}
		result = temp;
	}
}
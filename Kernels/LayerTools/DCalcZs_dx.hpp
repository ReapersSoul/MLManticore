#pragma once
#include <string>

static std::string DCalcZs_dx_Kernel_Code = R"(
void kernel DCalcZs_dx(global float* W, int WSizeX, int WSizeY, global float* Result, int ResultSize)
{
	int gid = get_global_id(0);
	int threads = get_global_size(0);
	//starting index for this thread
	int startingIndex = gid * (ResultSize / threads);
	//number of z's to calculate
	int Zs_to_calculate = ResultSize / threads;
	if (gid == threads - 1)
	{
		Zs_to_calculate += ResultSize % threads;
	}

	for (int i = startingIndex; i < startingIndex + Zs_to_calculate; i++)
	{
		Result[i] = 0;
		for (int j = 0; j < WSizeX; j++)
		{
			Result[i] += W[j * WSizeX + i];
		}
	}
}
)";
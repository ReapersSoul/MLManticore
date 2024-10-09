#pragma once
#include <string>

static std::string DCalcZs_dw_Kernel_Code = R"(
void kernel DCalcZs_dw(global float* X, int XSize, int WSizeX, int WSizeY, global float* Result, int ResultSizeX, int ResultSizeY)
{
	int gid = get_global_id(0);
	int threads = get_global_size(0);
	//starting index for this thread
	int startingIndex = gid * (ResultSizeX / threads);
	//number of z's to calculate
	int Zs_to_calculate = ResultSizeX / threads;
	if (gid == threads - 1)
	{
		Zs_to_calculate += ResultSizeX % threads;
	}

	for (int i = startingIndex; i < startingIndex + Zs_to_calculate; i++)
	{
		for (int j = 0; j < WSizeY; j++)
		{
			Result[i * WSizeY + j] = X[i];
		}
	}
}
)";
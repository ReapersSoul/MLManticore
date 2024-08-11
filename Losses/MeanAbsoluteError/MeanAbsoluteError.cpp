#include "MeanAbsoluteError.hpp"

double MeanAbsoluteError::Calculate(double output, double target){
	return abs(output-target);
}

double MeanAbsoluteError::Derivative(double output, double target){
	return output-target > 0 ? 1 : -1;
}

double MeanAbsoluteError::Calculate(std::vector<double> output, std::vector<double> target){
	double result;
	std::vector<double> temp;
	GPU_Sub(output, target, temp);
	GPU_Abs(temp, temp);
	GPU_Sum_Fast(temp, result);
	return result/output.size();
}

std::vector<double> MeanAbsoluteError::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result;
	GPU_MeanAbsoluteError_Derivative(output, target, result);
	return result;
}
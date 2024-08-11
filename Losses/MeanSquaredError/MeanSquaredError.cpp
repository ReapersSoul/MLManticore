#include "MeanSquaredError.hpp"

double MeanSquaredError::Calculate(double output, double target){
	return (output-target)*(output-target);
}

double MeanSquaredError::Derivative(double output, double target){
	return output-target;
}

double MeanSquaredError::Calculate(std::vector<double> output, std::vector<double> target){
	double result;
	std::vector<double> temp;
	GPU_Sub(output, target, temp);
	GPU_Mul(temp, temp, temp);
	GPU_Sum_Fast(temp, result);
	return result/output.size();
}

std::vector<double> MeanSquaredError::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result;
	GPU_Sub(output, target, result);
	GPU_Mul_Scalar(result, 2, result);
	return result;
}
#include "MeanAbsoluteError.hpp"

float MeanAbsoluteError::Calculate(float output, float target){
	return abs(output-target);
}

float MeanAbsoluteError::Derivative(float output, float target){
	return output-target > 0 ? 1 : -1;
}

float MeanAbsoluteError::Calculate(std::vector<float> output, std::vector<float> target){
	float result;
	std::vector<float> temp;
	Common::Sub(output, target, temp);
	Common::Abs(temp, temp);
	Common::Sum(temp, result);
	return result/output.size();
}

std::vector<float> MeanAbsoluteError::Derivative(std::vector<float> output, std::vector<float> target){
	std::vector<float> result;
	Common::MeanAbsoluteError_Derivative(output, target, result);
	return result;
}
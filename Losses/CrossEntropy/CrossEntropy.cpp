#include "CrossEntropy.hpp"

float CrossEntropy::Calculate(float output, float target){
	return -target*log(output);
}

float CrossEntropy::Derivative(float output, float target){
	return -target/output;
}

float CrossEntropy::Calculate(std::vector<float> output, std::vector<float> target){
	float result;
	Common::CrossEntropy(output, target, result);
	return result;
}

std::vector<float> CrossEntropy::Derivative(std::vector<float> output, std::vector<float> target){
	std::vector<float> result;
	Common::CrossEntropy_Derivative(output, target, result);
	return result;
}
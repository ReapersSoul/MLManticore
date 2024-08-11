#include "CrossEntropy.hpp"

double CrossEntropy::Calculate(double output, double target){
	return -target*log(output);
}

double CrossEntropy::Derivative(double output, double target){
	return -target/output;
}

double CrossEntropy::Calculate(std::vector<double> output, std::vector<double> target){
	double result;
	GPU_CrossEntropy(output, target, result);
	return result;
}

std::vector<double> CrossEntropy::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result;
	GPU_CrossEntropy_Derivative(output, target, result);
	return result;
}
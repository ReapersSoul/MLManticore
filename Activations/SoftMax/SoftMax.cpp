#include "SoftMax.hpp"

double SoftMax::Activate(double x){
	return exp(x);
}

double SoftMax::Derivative(double x){
	return x*(1-x);
}

std::vector<double> SoftMax::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_SoftMax(x, result);
	return result;
}

std::vector<double> SoftMax::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_SoftMax_Derivative(x, result);
	return result;
}

void SoftMax::Backward(double input, double fg, double lr){
	throw "Not implemented";
}

void SoftMax::Backward(std::vector<double> input, std::vector<double> fg, double lr){
	throw "Not implemented";
}

bool SoftMax::IsTrainable(){
	return false;
}
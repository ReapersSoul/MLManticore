#include "ReLU.hpp"

double ReLU::Activate(double x){
	return x > 0 ? x : 0;
}

double ReLU::Derivative(double x){
	return x > 0 ? 1 : 0;
}

std::vector<double> ReLU::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_ReLU(x, result);
	return result;
}

std::vector<double> ReLU::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_ReLU_Derivative(x, result);
	return result;
}

void ReLU::Backward(double input, double fg, double lr){
	throw "Not implemented";
}

void ReLU::Backward(std::vector<double> input, std::vector<double> fg, double lr){
	throw "Not implemented";
}

bool ReLU::IsTrainable(){
	return false;
}
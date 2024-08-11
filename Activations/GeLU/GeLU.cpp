#include "GeLU.hpp"

double GeLU::Activate(double x){
	return 0.5*x*(1+erf(x/sqrt(2)));
}

double GeLU::Derivative(double x){
	return 0.5*(1+erf(x/sqrt(2)) + x*exp(-x*x/2)/sqrt(2*M_PI));
}

std::vector<double> GeLU::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_GeLU(x, result);
	return result;
}

std::vector<double> GeLU::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_GeLU_Derivative(x, result);
	return result;
}

void GeLU::Backward(double input, double fg, double lr){
	throw "Not implemented";
}

void GeLU::Backward(std::vector<double> input, std::vector<double> fg, double lr){
	throw "Not implemented";
}

bool GeLU::IsTrainable(){
	return false;
}
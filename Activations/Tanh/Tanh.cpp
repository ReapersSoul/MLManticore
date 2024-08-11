#include "Tanh.hpp"

double Tanh::Activate(double x){
	return tanh(x);
}

double Tanh::Derivative(double x){
	return 1 - x*x;
}

std::vector<double> Tanh::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Tanh(x, result);
	return result;
}

std::vector<double> Tanh::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Tanh_Derivative(x, result);
	return result;
}

void Tanh::Backward(double input, double fg, double lr){
	throw "Not implemented";
}

void Tanh::Backward(std::vector<double> input, std::vector<double> fg, double lr){
	throw "Not implemented";
}

bool Tanh::IsTrainable(){
	return false;
}
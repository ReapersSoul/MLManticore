#include "Sigmoid.hpp"

double Sigmoid::Activate(double x){
	return 1/(1+exp(-x));
}

double Sigmoid::Derivative(double x){
	return exp(-x)/((1+exp(-x))*(1+exp(-x)));
}

std::vector<double> Sigmoid::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Sigmoid(x, result);
	return result;
}

std::vector<double> Sigmoid::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Sigmoid_Derivative(x, result);
	return result;
}

void Sigmoid::Backward(double input, double fg, double lr){
	throw "Not implemented";
}

void Sigmoid::Backward(std::vector<double> input, std::vector<double> fg, double lr){
	throw "Not implemented";
}

bool Sigmoid::IsTrainable(){
	return false;
}
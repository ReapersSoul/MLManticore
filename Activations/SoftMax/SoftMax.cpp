#include "SoftMax.hpp"

float SoftMax::Activate(float x){
	return exp(x);
}

float SoftMax::Derivative(float x){
	return x*(1-x);
}

std::vector<float> SoftMax::Activate(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::SoftMax(x, result);
	return result;
}

std::vector<float> SoftMax::Derivative(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::SoftMax_Derivative(x, result);
	return result;
}

void SoftMax::Backward(float input, float fg, float lr){
	throw "Not implemented";
}

void SoftMax::Backward(std::vector<float> input, std::vector<float> fg, float lr){
	throw "Not implemented";
}

bool SoftMax::IsTrainable(){
	return false;
}
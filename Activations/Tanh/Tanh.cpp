#include "Tanh.hpp"

float Tanh::Activate(float x){
	return tanh(x);
}

float Tanh::Derivative(float x){
	return 1 - x*x;
}

std::vector<float> Tanh::Activate(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::Tanh(x, result);
	return result;
}

std::vector<float> Tanh::Derivative(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::Tanh_Derivative(x, result);
	return result;
}

std::vector<std::vector<float>> Tanh::Activate(std::vector<std::vector<float>> x){
	std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
	for(int i = 0; i < x.size(); i++){
		Common::Tanh(x[i], result[i]);
	}
	return result;
}

std::vector<std::vector<float>> Tanh::Derivative(std::vector<std::vector<float>> x){
	std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
	for(int i = 0; i < x.size(); i++){
		Common::Tanh_Derivative(x[i], result[i]);
	}
	return result;
}

void Tanh::Backward(float input, float fg, float lr){
	throw "Not implemented";
}

void Tanh::Backward(std::vector<float> input, std::vector<float> fg, float lr){
	throw "Not implemented";
}

bool Tanh::IsTrainable(){
	return false;
}
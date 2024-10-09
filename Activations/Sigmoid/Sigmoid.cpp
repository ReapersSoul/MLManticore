#include "Sigmoid.hpp"

float Sigmoid::Activate(float x){
	return 1/(1+exp(-x));
}

float Sigmoid::Derivative(float x){
	return exp(-x)/((1+exp(-x))*(1+exp(-x)));
}

std::vector<float> Sigmoid::Activate(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::Sigmoid(x, result);
	return result;
}

std::vector<float> Sigmoid::Derivative(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::Sigmoid_Derivative(x, result);
	return result;
}

std::vector<std::vector<float>> Sigmoid::Activate(std::vector<std::vector<float>> x){
  std::vector<std::vector<float>> result(x.size(),std::vector<float>(x[0].size()));
  for (int i = 0; i < x.size(); i++) {
    Common::Sigmoid(x[i], result[i]);
  }
  return result;
}

std::vector<std::vector<float>> Sigmoid::Derivative(std::vector<std::vector<float>> x){
  std::vector<std::vector<float>> result(x.size(),std::vector<float>(x[0].size()));
  for (int i = 0; i < x.size(); i++) {
    Common::Sigmoid_Derivative(x[i], result[i]);
  }
  return result;
}

void Sigmoid::Backward(float input, float fg, float lr){
	throw "Not implemented";
}

void Sigmoid::Backward(std::vector<float> input, std::vector<float> fg, float lr){
	throw "Not implemented";
}

bool Sigmoid::IsTrainable(){
	return false;
}
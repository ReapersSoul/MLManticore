#include "GeLU.hpp"

float GeLU::Activate(float x){
	return 0.5*x*(1+erf(x/sqrt(2)));
}

float GeLU::Derivative(float x){
	return 0.5*(1+erf(x/sqrt(2)) + x*exp(-x*x/2)/sqrt(2*M_PI));
}

std::vector<float> GeLU::Activate(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::GeLU(x, result);
	return result;
}

std::vector<float> GeLU::Derivative(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::GeLU_Derivative(x, result);
	return result;
}

std::vector<std::vector<float>> GeLU::Activate(std::vector<std::vector<float>> input) {
  std::vector<std::vector<float>> result(input.size());
  for (int i = 0; i < input.size(); i++) {
    Common::GeLU(input[i], result[i]);
  }
  return result;
}

std::vector<std::vector<float>> GeLU::Derivative(std::vector<std::vector<float>> input){
  std::vector<std::vector<float>> result(input.size());
  for (int i = 0; i < input.size(); i++) {
    Common::GeLU_Derivative(input[i], result[i]);
  }
  return result;
}

void GeLU::Backward(float input, float fg, float lr){
	throw "Not implemented";
}

void GeLU::Backward(std::vector<float> input, std::vector<float> fg, float lr){
	throw "Not implemented";
}

bool GeLU::IsTrainable(){
	return false;
}
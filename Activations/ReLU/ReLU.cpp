#include "ReLU.hpp"

float ReLU::Activate(float x){
	return x > 0 ? x : 0;
}

float ReLU::Derivative(float x){
	return x > 0 ? 1 : 0;
}

std::vector<float> ReLU::Activate(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::ReLU(x, result);
	return result;
}

std::vector<float> ReLU::Derivative(std::vector<float> x){
	std::vector<float> result(x.size());
	Common::ReLU_Derivative(x, result);
	return result;
}

std::vector<std::vector<float>> ReLU::Activate(std::vector<std::vector<float>> x){
  std::vector<std::vector<float>> result(x.size());
  for(int i = 0; i < x.size(); i++){
    Common::ReLU(x[i], result[i]);
  }
  return result;
}

std::vector<std::vector<float>> ReLU::Derivative(std::vector<std::vector<float>> x){
  std::vector<std::vector<float>> result(x.size());
  for(int i = 0; i < x.size(); i++){
    Common::ReLU_Derivative(x[i], result[i]);
  }
  return result;
}

void ReLU::Backward(float input, float fg, float lr){
	throw "Not implemented";
}

void ReLU::Backward(std::vector<float> input, std::vector<float> fg, float lr){
	throw "Not implemented";
}

bool ReLU::IsTrainable(){
	return false;
}
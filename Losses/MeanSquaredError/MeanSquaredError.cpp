#include "MeanSquaredError.hpp"

float MeanSquaredError::Calculate(float output, float target){
	return (output-target)*(output-target);
}

float MeanSquaredError::Derivative(float output, float target){
	return output-target;
}

float MeanSquaredError::Calculate(std::vector<float> output, std::vector<float> target){
	// Common::Sub(output, target, temp);
	// Common::Mul(temp, temp, temp);
	// Common::Sum_Fast(temp, result);
  float result=0;
  for (int i = 0; i < output.size(); i++) {
    result+=(output[i]-target[i])*(output[i]-target[i]);
  }
	return result/output.size();
}

std::vector<float> MeanSquaredError::Derivative(std::vector<float> output, std::vector<float> target){
	std::vector<float> result;
	// Common::Sub(output, target, result);
	// Common::Mul_Scalar(result, 2, result);
  for (int i = 0; i < output.size(); i++) {
    result.push_back((output[i]-target[i])*2);
  }
	return result;
}
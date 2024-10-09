#include "LogLoss.hpp"

float LogLoss::Calculate(float output, float target){
	return -target*log(output) - (1-target)*log(1-output);
}

float LogLoss::Derivative(float output, float target){
	return (output-target)/(output*(1-output));
}

float LogLoss::Calculate(std::vector<float> output, std::vector<float> target){
	float result;
	Common::LogLoss(output, target, result);
	return result;
}

std::vector<float> LogLoss::Derivative(std::vector<float> output, std::vector<float> target){
	std::vector<float> result;
	Common::LogLoss_Derivative(output, target, result);
	return result;
}
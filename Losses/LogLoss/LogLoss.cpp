#include "LogLoss.hpp"

double LogLoss::Calculate(double output, double target){
	return -target*log(output) - (1-target)*log(1-output);
}

double LogLoss::Derivative(double output, double target){
	return (output-target)/(output*(1-output));
}

double LogLoss::Calculate(std::vector<double> output, std::vector<double> target){
	double result;
	GPU_LogLoss(output, target, result);
	return result;
}

std::vector<double> LogLoss::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result;
	GPU_LogLoss_Derivative(output, target, result);
	return result;
}
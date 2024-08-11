#pragma once
#include <Losses/LossFunction.hpp>

class MeanAbsoluteError : public LossFunction
{
public:
	double Calculate(double output, double target);
	double Derivative(double output, double target);
	double Calculate(std::vector<double> output, std::vector<double> target);
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target);
};
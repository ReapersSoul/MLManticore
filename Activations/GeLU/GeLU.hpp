#pragma once
#include <Activations/ActivationFunction.hpp>

class GeLU : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
	void Backward(double input, double fg, double lr);
	void Backward(std::vector<double> input, std::vector<double> fg, double lr);
	bool IsTrainable();
};
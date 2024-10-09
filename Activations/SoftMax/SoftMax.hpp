#pragma once
#include <Common.hpp>
#include <Activations/ActivationFunction.hpp>
#include <cmath>

class SoftMax : public ActivationFunction
{
public:
	float Activate(float input);
	float Derivative(float input);
	std::vector<float> Activate(std::vector<float> input);
	std::vector<float> Derivative(std::vector<float> input);
	
	void Backward(float input, float fg, float lr);
	void Backward(std::vector<float> input, std::vector<float> fg, float lr);
	bool IsTrainable();
};
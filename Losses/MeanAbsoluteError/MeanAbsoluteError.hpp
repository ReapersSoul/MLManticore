#pragma once
#include <Losses/LossFunction.hpp>

class MeanAbsoluteError : public LossFunction
{
public:
	float Calculate(float output, float target);
	float Derivative(float output, float target);
	float Calculate(std::vector<float> output, std::vector<float> target);
	std::vector<float> Derivative(std::vector<float> output, std::vector<float> target);
};
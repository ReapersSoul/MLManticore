#pragma once
#include <Activations/ActivationFunction.hpp>

class LeakyReLU : public ActivationFunction
{
public:
	float alpha;
	LeakyReLU();
	LeakyReLU(float Alpha);
	float Activate(float input) override;
	float Derivative(float input) override;
	std::vector<float> Activate(std::vector<float> input) override;
	std::vector<float> Derivative(std::vector<float> input) override;
	std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> input) override;
	std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> input) override;
	std::vector<std::vector<std::vector<float>>> Activate(std::vector<std::vector<std::vector<float>>> input) override;
	std::vector<std::vector<std::vector<float>>> Derivative(std::vector<std::vector<std::vector<float>>> input) override;
};
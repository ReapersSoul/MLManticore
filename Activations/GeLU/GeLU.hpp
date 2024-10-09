#pragma once
#include <Activations/ActivationFunction.hpp>
#define M_PI 3.14159265358979323846

class GeLU : public ActivationFunction
{
public:
	float Activate(float input);
	float Derivative(float input);
	std::vector<float> Activate(std::vector<float> input);
	std::vector<float> Derivative(std::vector<float> input);
  std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> input);
  std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> input);
	void Backward(float input, float fg, float lr);
	void Backward(std::vector<float> input, std::vector<float> fg, float lr);
	bool IsTrainable();
};
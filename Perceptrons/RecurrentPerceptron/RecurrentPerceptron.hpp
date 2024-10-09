#pragma once
#include <vector>
#include <functional>

#include <Perceptrons/Perceptron.hpp>
#include <Activations/ActivationFunction.hpp>

class RecurrentPerceptron: public Perceptron
{
private:
	int InSize;
	std::vector<float> w,x;
	float b,z,last_activation;
public:
	void Init(int InSize, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1);

	void Resize(int InSize, float min, float max);

	float Forward(std::vector<float> x);

	std::vector<float> Backward(float fg,float lr=0.01);

	void RandomizeWeights(float min, float max);

	void RandomizeBias(float min, float max);
};
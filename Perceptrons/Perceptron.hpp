#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>

class Perceptron
{
protected:
	ActivationFunction* af;
	float min_clamp;
	float max_clamp;
public:
	Perceptron();
	~Perceptron();

	virtual void Init(int InSize, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1)=0;

	virtual void Resize(int InSize, float min, float max)=0;

	virtual float Forward(std::vector<float> x)=0;
	virtual std::vector<float> Backward(float fg,float lr=0.01)=0;

	virtual void RandomizeWeights(float min, float max)=0;
	virtual void RandomizeBias(float min, float max)=0;
};
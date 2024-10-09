#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>

class OneDLayer
{
private:
	ActivationFunction* af;
	float min_clamp;
	float max_clamp;
public:
	OneDLayer();
	~OneDLayer();

	virtual void Init(int InSize, int OutSize, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1)=0;

	virtual void Resize(int InSize, int OutSize, float min, float max)=0;

	virtual std::vector<float> Forward(std::vector<float> x)=0;
	virtual std::vector<float> Backward(std::vector<float> fg,float lr=0.01)=0;

	virtual void RandomizeWeights(float min, float max)=0;
	virtual void RandomizeBias(float min, float max)=0;
};
#pragma once
#include <vector>
#include <functional>
#include <glm/glm.hpp>

#include <Activations/ActivationFunction.hpp>

class NDLayer
{
private:
	ActivationFunction* af;
	float min_clamp;
	float max_clamp;
public:
	NDLayer();
	~NDLayer();

	virtual void Init(std::vector<int> InShape, std::vector<int> OutShape, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1)=0;

	virtual void Resize(std::vector<int> InShape, std::vector<int> OutShape, float min, float max)=0;

	virtual std::vector<float> Forward(std::vector<float> x)=0;
	virtual std::vector<float> Backward(std::vector<float> fg,float lr=0.01)=0;

	virtual void RandomizeWeights(float min, float max)=0;
	virtual void RandomizeBias(float min, float max)=0;
};
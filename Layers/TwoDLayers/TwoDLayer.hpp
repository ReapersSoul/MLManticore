#pragma once
#include <vector>
#include <functional>
#include <glm/glm.hpp>

#include <Activations/ActivationFunction.hpp>

class TwoDLayer
{
private:
	ActivationFunction* af;
	float min_clamp;
	float max_clamp;
public:
	TwoDLayer();
	~TwoDLayer();

	virtual void Init(glm::ivec2 InSize, glm::ivec2 OutSize, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1)=0;

	virtual void Resize(glm::ivec2 InSize, glm::ivec2 OutSize, float min, float max)=0;

	virtual std::vector<std::vector<float>> Forward(std::vector<std::vector<float>> x)=0;
	virtual std::vector<std::vector<float>> Backward(std::vector<std::vector<float>> fg,float lr=0.01)=0;

	virtual void RandomizeWeights(float min, float max)=0;
	virtual void RandomizeBias(float min, float max)=0;
};
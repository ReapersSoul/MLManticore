#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>

class RecurrentLayer
{
private:
	std::vector<std::vector<float>> w;
	std::vector<float> b,z,x,activationHistory;
	ActivationFunction* af;
	float min,max,min_clamp,max_clamp;
public:
	RecurrentLayer();
	~RecurrentLayer();

	void Init(int InSize, int OutSize, ActivationFunction* af, float min=-1.0, float max=1.0, float min_clamp=-1.0, float max_clamp=1.0);

	std::vector<float> Forward(std::vector<float> x);
	std::vector<float> Backward(std::vector<float> fg=std::vector<float>(1),float lr=0.01);

	std::vector<std::vector<float>> GetWeights();
	void SetWeights(std::vector<std::vector<float>> w);
	void RandomizeWeights(float min, float max);
	void ResizeWithRandomForNewWeights(int InSize, int OutSize, float min, float max);

	std::vector<float> GetBias();
	void SetBias(std::vector<float> b);
	void RandomizeBias(float min, float max);

	std::vector<float> GetX();
	std::vector<float> GetZ();
	std::vector<float> GetPreviousActivation();
};
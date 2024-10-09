#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>

class BasicLayer
{
private:
	std::vector<std::vector<float>> w,dw;
	std::vector<float> b,z,x,db;
	ActivationFunction* af;
	float min_clamp;
	float max_clamp;
public:
	BasicLayer();
	~BasicLayer();

	void Init(int InSize, int OutSize, ActivationFunction* af, float min=-1.0, float max=1.0,float min_clamp=-1,float max_clamp=1);

	void Resize(int InSize, int OutSize, float min, float max);

	std::vector<float> Forward(std::vector<float> x);
	std::vector<float> Backward(std::vector<float> fg=std::vector<float>(1),float lr=0.01);

	std::vector<float> GetWeights();
	void SetWeights(std::vector<std::vector<float>> w);
	void RandomizeWeights(float min, float max);
	void ResizeWithRandomForNewWeights(int InSize,int OutSize, float min, float max);

	std::vector<float> GetBias();
	void SetBias(std::vector<float> b);
	void RandomizeBias(float min, float max);

	std::vector<std::vector<float>> GetDWeights();
	std::vector<float> GetDBias();
};
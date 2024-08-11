#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>
#include <Activations/SoftMax/SoftMax.hpp>
#include <Models/PerceptronLayer/PerceptronLayer.hpp>

class MambaLayer
{
private:
	std::vector<std::vector<double>> w,gw;
	std::vector<double> b,z,x,gx;
	ActivationFunction* af;
		int inputs;
	int outputs;
	int w_size;

	PerceptronLayer pl;

	std::vector<std::vector<double>> Split(std::vector<double> vec, int width);
	std::vector<double> Flatten(std::vector<std::vector<double>> vec);

public:
	MambaLayer();
	~MambaLayer();

	void Init(int InSize, int OutSize, ActivationFunction* af, double min=-1.0, double max=1.0);

	std::vector<double> Forward(std::vector<double> x);
	std::vector<double> Backward(std::vector<double> fg=std::vector<double>(1),double lr=0.01);

	std::vector<double> GetGeneratorWeights();
	void SetWeights(std::vector<std::vector<double>> w);
	void RandomizeGeneratorWeights(double min, double max);
	void ResizeWithRandomForNewGeneratorWeights(int InSize,int OutSize, double min, double max);

	std::vector<double> GetBias();
	void SetBias(std::vector<double> b);
	void RandomizeBias(double min, double max);
};
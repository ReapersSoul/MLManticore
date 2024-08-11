#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>
#include <Activations/SoftMax/SoftMax.hpp>
#include <Models/PerceptronLayer/PerceptronLayer.hpp>
#include <tracy/Tracy.hpp>

class MambaPerceptron
{
private:
	std::vector<double> x,gx,w;
	double b,z;
	ActivationFunction* af;
	int inputs;
	int outputs;
	int w_size;

	PerceptronLayer pl;
public:
	MambaPerceptron();
	~MambaPerceptron();

	void Init(int InSize, ActivationFunction* af, double min=-1.0, double max=1.0);

	double Forward(std::vector<double> x);
	std::vector<double> Backward(double fg=1,double lr=0.01);

	std::vector<double> GetGeneratorWeights();
	void SetWeights(std::vector<double> w);
	void RandomizeGeneratorWeights(double min, double max);
	void ResizeWithRandomForNewGeneratorWeights(int size, double min, double max);

	double GetBias();
	void SetBias(double b);
	void RandomizeBias(double min, double max);

	std::vector<double> GetX();
	double GetZ();
};
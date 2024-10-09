#include "DeepNeuralNetwork.hpp"

DeepNeuralNetwork::DeepNeuralNetwork()
{
}

DeepNeuralNetwork::~DeepNeuralNetwork()
{
}

void DeepNeuralNetwork::Init(ActivationFunction *AF, std::vector<int> layerSizes, float min, float max, float clamp_min, float clamp_max)
{
	layers.resize(layerSizes.size() - 1);
	for (int i = 0; i < layerSizes.size() - 1; i++)
	{
		layers[i].Init(layerSizes[i], layerSizes[i + 1], AF, min, max, clamp_min, clamp_max);
	}
}

void DeepNeuralNetwork::resize(std::vector<int> layerSizes, float min, float max)
{
	layers.resize(layerSizes.size() - 1);
	for (int i = 0; i < layerSizes.size() - 1; i++)
	{
		layers[i].Resize(layerSizes[i], layerSizes[i + 1], min, max);
	}
}

std::vector<float> DeepNeuralNetwork::Forward(std::vector<float> input)
{
	std::vector<float> output = input;
	for (int i = 0; i < layers.size(); i++)
	{
		output = layers[i].Forward(output);
	}
	return output;
}

std::vector<float> DeepNeuralNetwork::Backward(std::vector<float> fg, float lr)
{
	std::vector<float> output = fg;
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		output = layers[i].Backward(output, lr);
	}
	return output;
}

void DeepNeuralNetwork::RandomizeWeights(float min, float max)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].RandomizeWeights(min, max);
	}
}

void DeepNeuralNetwork::RandomizeBias(float min, float max)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i].RandomizeBias(min, max);
	}
}

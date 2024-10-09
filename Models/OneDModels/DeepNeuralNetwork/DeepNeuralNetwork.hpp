#pragma once
#include <vector>
#include <functional>
#include <thread>

#include <Common.hpp>
#include <Layers/OneDLayers/BasicLayer/BasicLayer.hpp>

class DeepNeuralNetwork
{
private:
	std::vector<BasicLayer> layers;

public:
	DeepNeuralNetwork();
	~DeepNeuralNetwork();

	void Init(ActivationFunction *AF, std::vector<int> layerSizes, float min, float max, float clamp_min, float clamp_max);

	void resize(std::vector<int> layerSizes, float min, float max);

	std::vector<float> Forward(std::vector<float> input);

	std::vector<float> Backward(std::vector<float> fg, float lr);

	void RandomizeWeights(float min, float max);

	void RandomizeBias(float min, float max);
};
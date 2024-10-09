#pragma once
#include <vector>
#include <functional>
#include <thread>
#include <MLManticore.hpp>
#include <Activations/ActivationFunction.hpp>

class RecurrentNeuralNetwork
{
private:
	std::vector<std::vector<std::vector<float>>> weights, activationHistory;
	std::vector<std::vector<float>> biases, preActivation, input;
	ActivationFunction *af;
	std::vector<int> LayerSizes;
	int memoryLength;

	static float clamp(float value, float min, float max);

	static float validate(float value, float default_value);

    static void resizeLayer(int inputSize, int outputSize, int memoryLength, std::vector<float> &input, std::vector<std::vector<float>> &OutputHistory, std::vector<std::vector<float>> &weights, std::vector<float> &biases, std::vector<float>& preActivations, float min, float max);

    static std::vector<float> forwardLayer(int memoryLength, std::vector<float> input, std::vector<std::vector<float>> &OutputHistory, std::vector<std::vector<float>> weights, std::vector<float> preActivation, std::vector<float> biases, ActivationFunction* af);

    static std::vector<float> backwardLayer(int memoryLength, std::vector<float> fg, std::vector<float> input, std::vector<std::vector<float>> &OutputHistory, std::vector<std::vector<float>> &weights, std::vector<float> &biases, std::vector<float> preActivation, ActivationFunction* af, float lr);

public:
	RecurrentNeuralNetwork();
	~RecurrentNeuralNetwork();

	void resizeNet(std::vector<int> LayerSizes, int contextSize, float min, float max);

	void init(std::vector<int> LayerSizes, int contextSize, ActivationFunction *af, float min = -.00001, float max = .00001);

	std::vector<float> forward(std::vector<float> x);
	std::vector<float> backward(std::vector<float> fg , float lr = 0.01);

	std::vector<std::vector<std::vector<float>>> getWeights();
	void setWeights(std::vector<std::vector<std::vector<float>>> w);
	void randomizeWeights(float min, float max);
	void resizeWithRandomForNewWeights(int InSize, int OutSize, float min, float max);

  std::vector<int> getLayerSizes();
	std::vector<std::vector<float>> getBiases();
	void setBiases(std::vector<std::vector<float>> b);
	void randomizeBiases(float min, float max);
	void setActivationFunction(ActivationFunction *af);
	void setMemoryLength(int ContextSize, float min, float max);
	int getMemoryLength();

	// getters
	std::vector<std::vector<std::vector<float>>> getW();
	std::vector<std::vector<float>> getB();
	std::vector<std::vector<float>> getZ();
	std::vector<std::vector<float>> getX();
	std::vector<std::vector<float>> getPreviousActivation();
};
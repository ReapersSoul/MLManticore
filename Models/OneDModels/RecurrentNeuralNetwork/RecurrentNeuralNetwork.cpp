#include "RecurrentNeuralNetwork.hpp"

RecurrentNeuralNetwork::RecurrentNeuralNetwork()
{
}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{
    delete af;
}

float RecurrentNeuralNetwork::clamp(float value, float min, float max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

float RecurrentNeuralNetwork::validate(float value, float default_value)
{
	// if nan return the default value
	if (value != value)
	{
		return default_value;
	}
	// if inf return the default value
	if (value != value || value == std::numeric_limits<float>::infinity() || value == -std::numeric_limits<float>::infinity())
	{
		return default_value;
	}
	// if the value is less than the minimum value return the minimum value
	if (value < std::numeric_limits<float>::lowest())
	{
		return std::numeric_limits<float>::lowest();
	}
	// if the value is greater than the maximum value return the maximum value
	if (value > std::numeric_limits<float>::max())
	{
		return std::numeric_limits<float>::max();
	}
	// if 0 return the closest non-zero value
	if (value == 0)
	{
		return .0000000001;
	}
	// otherwise return the value
	return value;
}

void RecurrentNeuralNetwork::resizeLayer(int inputSize, int outputSize, int memoryLength, std::vector<float>& input, std::vector<std::vector<float>>& OutputHistory, std::vector<std::vector<float>>& weights, std::vector<float>& biases, std::vector<float>& preActivation, float min, float max){
	input.resize(inputSize, 0.0);
	OutputHistory.resize(memoryLength, std::vector<float>(outputSize, 0.0));
	int trueInputSize = inputSize + (outputSize * memoryLength);
	weights.resize(outputSize, std::vector<float>(trueInputSize, Common::RandRange(min, max)));
	biases.resize(outputSize, Common::RandRange(min, max));
	preActivation.resize(outputSize, Common::RandRange(min, max));
}

void RecurrentNeuralNetwork::resizeNet(std::vector<int> LayerSizes, int memoryLength, float min, float max)
{
    this->LayerSizes = LayerSizes;
    this->memoryLength = memoryLength;
    input.resize(LayerSizes.size());
    activationHistory.resize(LayerSizes.size());
    weights.resize(LayerSizes.size()-1);
    biases.resize(LayerSizes.size()-1);
    preActivation.resize(LayerSizes.size()-1);
	for (int i = 0; i < LayerSizes.size()-1; i++)
	{
		resizeLayer(LayerSizes[i], LayerSizes[i + 1], memoryLength, input[i], activationHistory[i], weights[i], biases[i], preActivation[i],min,max);
	}
}

void RecurrentNeuralNetwork::init(std::vector<int> LayerSizes, int memoryLength, ActivationFunction *af, float min, float max)
{
	this->af = af;

	resizeNet(LayerSizes, memoryLength, min, max);

	randomizeWeights(min, max);
	randomizeBiases(min, max);
}

std::vector<float> RecurrentNeuralNetwork::forwardLayer(int memoryLength, std::vector<float> input, std::vector<std::vector<float>>& OutputHistory, std::vector<std::vector<float>> weights, std::vector<float> preActivation, std::vector<float> biases, ActivationFunction *af) {
	std::vector<float> trueInput;
	trueInput.insert(trueInput.end(), input.begin(), input.end());
	for (int i = 0; i < memoryLength; i++) {
		trueInput.insert(trueInput.end(), OutputHistory[i].begin(), OutputHistory[i].end());
	}
	std::vector<float> output;
//	for (int i = 0; i < weights.size(); i++) {
//		float sum = 0;
//		for (int j = 0; j < weights[i].size(); j++) {
//			sum += weights[i][j] * trueInput[j];
//		}
//		output.push_back(sum + biases[i]);
//	}

    Common::CalcZs(trueInput, weights, biases, output);

	preActivation = output;
	output = af->Activate(output);
	OutputHistory.push_back(output);
	if (OutputHistory.size() > memoryLength) {
		OutputHistory.erase(OutputHistory.begin());
	}
	return output;
}

std::vector<float> RecurrentNeuralNetwork::backwardLayer(int memoryLength, std::vector<float> fg, std::vector<float> input, std::vector<std::vector<float>> &OutputHistory, std::vector<std::vector<float>> &weights, std::vector<float> &biases, std::vector<float> preActivation, ActivationFunction* af, float lr){
	std::vector<float> trueInput;
	trueInput.insert(trueInput.end(), input.begin(), input.end());
	for (int i = 0; i < memoryLength; i++) {
		trueInput.insert(trueInput.end(), OutputHistory[i].begin(), OutputHistory[i].end());
	}
	std::vector<float> dLdA;
	for (int i = 0; i < fg.size(); i++) {
		dLdA.push_back(fg[i] * af->Derivative(preActivation[i]));
	}
	std::vector<float> dLdW;
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			dLdW.push_back(dLdA[i] * trueInput[j]);
		}
	}
	std::vector<float> dLdB;
	for (int i = 0; i < biases.size(); i++) {
		dLdB.push_back(dLdA[i]);
	}
	std::vector<float> dLdX;
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			dLdX.push_back(dLdA[i] * weights[i][j]);
		}
	}
	std::vector<float> dLdXInput;
	for (int i = 0; i < input.size(); i++) {
		dLdXInput.push_back(dLdX[i]);
	}
	std::vector<float> dLdXOutputHistory;
	for (int i = 0; i < memoryLength; i++) {
		for (int j = 0; j < OutputHistory[i].size(); j++) {
			dLdXOutputHistory.push_back(dLdX[i + j]);
		}
	}
	std::vector<float> dLdXOutput;
	for (int i = 0; i < memoryLength; i++) {
		dLdXOutput.push_back(dLdXOutputHistory[i]);
	}
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			weights[i][j] -= lr * dLdW[i + j];
		}
	}
	for (int i = 0; i < biases.size(); i++) {
		biases[i] -= lr * dLdB[i];
	}
	return dLdXInput;
}

std::vector<float> RecurrentNeuralNetwork::forward(std::vector<float> input)
{
	// forward the network layer by layer using ForwardLayer
	// return the output of the last layer
	this->input[0] = input;
	for (int i = 0; i < weights.size(); i++)
	{
        this->input[i + 1] = forwardLayer(memoryLength, this->input[i],  activationHistory[i], weights[i], preActivation[i], biases[i], af);
	}

	return this->input[weights.size()];
}

std::vector<float> RecurrentNeuralNetwork::backward(std::vector<float> fg, float lr)
{
	// backward the network layer by layer using BackwardLayer
	// return the output of the first layer
	for (int i = weights.size() - 1; i >= 0; i--)
	{
		fg = backwardLayer(memoryLength, fg, this->input[i], activationHistory[i], weights[i], biases[i], preActivation[i], af, lr);
	}
	return fg;
}

std::vector<std::vector<std::vector<float>>> RecurrentNeuralNetwork::getWeights()
{
	return weights;
}

void RecurrentNeuralNetwork::setWeights(std::vector<std::vector<std::vector<float>>> weights)
{
	weights = weights;
}

void RecurrentNeuralNetwork::randomizeWeights(float min, float max)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				float r = Common::RandRange(min, max);
				while (r == 0.0)
					r = Common::RandRange(min, max);
				weights[i][j][k] = r;
			}
		}
	}
}

void RecurrentNeuralNetwork::resizeWithRandomForNewWeights(int InSize, int OutSize, float min, float max)
{
	int wsize = weights.size();
	weights.resize(InSize);
	for (int i = wsize; i < weights.size(); i++)
	{
		weights[i].resize(OutSize);
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j].resize(OutSize);
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				float r = Common::RandRange(min, max);
				while (r == 0.0)
					r = Common::RandRange(min, max);
				weights[i][j][k] = r;
			}
		}
	}
}

std::vector<std::vector<float>> RecurrentNeuralNetwork::getBiases()
{
	return biases;
}

void RecurrentNeuralNetwork::setBiases(std::vector<std::vector<float>> bias)
{
	biases = bias;
}

void RecurrentNeuralNetwork::randomizeBiases(float min, float max)
{
	for (int j = 0; j < biases.size(); j++)
	{
		for (int k = 0; k < biases[j].size(); k++)
		{
			float r = Common::RandRange(min, max);
			while (r == 0.0)
				r = Common::RandRange(min, max);
			biases[j][k] = r;
		}
	}
}

std::vector<std::vector<std::vector<float>>> RecurrentNeuralNetwork::getW()
{
	return weights;
}

std::vector<std::vector<float>> RecurrentNeuralNetwork::getB()
{
	return biases;
}

std::vector<std::vector<float>> RecurrentNeuralNetwork::getZ()
{
	return preActivation;
}

std::vector<std::vector<float>> RecurrentNeuralNetwork::getX()
{
	return input;
}

std::vector<std::vector<float>> RecurrentNeuralNetwork::getPreviousActivation()
{
	return activationHistory[activationHistory.size() - 1];
}

std::vector<int> RecurrentNeuralNetwork::getLayerSizes()
{
    return LayerSizes;
}

void RecurrentNeuralNetwork::setActivationFunction(ActivationFunction *af)
{
    if(this->af != nullptr)
        delete this->af;
	this->af = af;
}

void RecurrentNeuralNetwork::setMemoryLength(int memoryLength, float min, float max)
{
	this->memoryLength = memoryLength;
	resizeNet(LayerSizes, memoryLength, min, max);
}

int RecurrentNeuralNetwork::getMemoryLength()
{
	return memoryLength;
}
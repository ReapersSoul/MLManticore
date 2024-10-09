#include <Models/OneDModels/MambaNeuralNetwork/MambaNeuralNetwork.hpp>

void MambaNeuralNetwork::resize(std::vector<int> layerSizes,float min, float max) {
  layers.resize(layerSizes.size());
  for (int i = 0; i < layers.size(); i++) {
    layers[i].ResizeWithRandomForNewGeneratorWeights(layerSizes[i],layerSizes[i+1],min,max);
  }
}

void MambaNeuralNetwork::init(std::vector<int> layerSizes, ActivationFunction* af, float min, float max) {
  layers.resize(layerSizes.size());
  for (int i = 0; i < layers.size(); i++) {
    layers[i].Init(layerSizes[i],layerSizes[i+1],af,min,max);
  }
}

std::vector<float> MambaNeuralNetwork::forward(std::vector<float> x) {
  for (int i = 0; i < layers.size(); i++) {
    x=layers[i].Forward(x);
  }
  return x;
}

std::vector<float> MambaNeuralNetwork::backward(std::vector<float> fg,float lr) {
  for (int i = layers.size()-1; i >= 0; i--) {
    fg=layers[i].Backward(fg,lr);
  }
  return fg;
}

std::vector<std::vector<std::vector<float>>> MambaNeuralNetwork::getWeights() {
  std::vector<std::vector<std::vector<float>>> result;
  for (int i = 0; i < layers.size(); i++) {
    result.push_back(layers[i].GetGeneratorWeights());
  }
  return result;
}

void MambaNeuralNetwork::setWeights(std::vector<std::vector<std::vector<float>>> w) {
  for (int i = 0; i < layers.size(); i++) {
    layers[i].SetWeights(w[i]);
  }
}

std::vector<std::vector<float>> MambaNeuralNetwork::getBiases() {
  std::vector<std::vector<float>> result;
  for (int i = 0; i < layers.size(); i++) {
    result.push_back(layers[i].GetBias());
  }
  return result;
}

void MambaNeuralNetwork::setBiases(std::vector<std::vector<float>> b) {
  for (int i = 0; i < layers.size(); i++) {
    layers[i].SetBias(b[i]);
  }
}

void MambaNeuralNetwork::randomizeWeights(float min, float max) {
  for (int i = 0; i < layers.size(); i++) {
    layers[i].RandomizeGeneratorWeights(min,max);
  }
}
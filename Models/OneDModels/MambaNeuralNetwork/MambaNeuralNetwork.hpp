#pragma once
#include <vector>
#include <Layers/OneDLayers/MambaLayer/MambaLayer.hpp>

class MambaNeuralNetwork
{
  std::vector<MambaLayer> layers;

public:

  void resize(std::vector<int> layerSizes,float min, float max);

  void init(std::vector<int> layerSizes, ActivationFunction* af, float min, float max);

  std::vector<float> forward(std::vector<float> x);
  std::vector<float> backward(std::vector<float> fg,float lr=0.01);

  std::vector<std::vector<std::vector<float>>> getWeights();
  void setWeights(std::vector<std::vector<std::vector<float>>> w);

  std::vector<std::vector<float>> getBiases();
  void setBiases(std::vector<std::vector<float>> b);

  void randomizeWeights(float min, float max);
  
};
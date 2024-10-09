#pragma once
#include <vector>
#include <functional>

#include <Activations/ActivationFunction.hpp>
#include <Activations/SoftMax/SoftMax.hpp>
#include <Layers/OneDLayers/BasicLayer/BasicLayer.hpp>

class MambaLayer
{
private:
  std::vector<std::vector<float>> w, gw;
  std::vector<float> b, z, x, gx;
  ActivationFunction *af;
  int inputs;
  int outputs;
  int w_size;

  BasicLayer bl;

  std::vector<std::vector<float>> Split(std::vector<float> vec, int width);
  std::vector<float> Flatten(std::vector<std::vector<float>> vec);

public:
  MambaLayer();
  ~MambaLayer();

  void Init(int InSize, int OutSize, ActivationFunction *af, float min = -1.0, float max = 1.0);

  std::vector<float> Forward(std::vector<float> x);
  std::vector<float> Backward(std::vector<float> fg = std::vector<float>(1), float lr = 0.01);

  std::vector<std::vector<float>> GetGeneratorWeights();
  void SetWeights(std::vector<std::vector<float>> w);
  void RandomizeGeneratorWeights(float min, float max);
  void ResizeWithRandomForNewGeneratorWeights(int InSize, int OutSize, float min, float max);

  std::vector<float> GetBias();
  void SetBias(std::vector<float> b);
  void RandomizeBias(float min, float max);
};
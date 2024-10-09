#include "LerpEndToEnd.hpp"

LerpEndToEnd::LerpEndToEnd()
{
  P = {0, 1, 2, 3};
}

LerpEndToEnd::LerpEndToEnd(std::vector<float> P)
{
  this->P = P;
}

LerpEndToEnd::LerpEndToEnd(int res, float min, float max)
{
  this->min = min;
  this->max = max;
  SetResolution(res, min, max);
}

LerpEndToEnd::~LerpEndToEnd()
{
}

float LerpEndToEnd::Activate(float input)
{
  return lerpn(sigmoid(input), P);
}

float LerpEndToEnd::Derivative(float input)
{
  return dlerpn_dt(sigmoid(input), P) * dsigmoid_dx(input);
}

std::vector<float> LerpEndToEnd::Activate(std::vector<float> input)
{
  for (int i = 0; i < input.size(); i++)
  {
    input[i] = Activate(input[i]);
  }
  return input;
}

std::vector<float> LerpEndToEnd::Derivative(std::vector<float> input)
{
  for (int i = 0; i < input.size(); i++)
  {
    input[i] = Derivative(input[i]);
  }
  return input;
}

std::vector<std::vector<float>> LerpEndToEnd::Activate(std::vector<std::vector<float>> input)
{
  for (int i = 0; i < input.size(); i++)
  {
    for (int j = 0; j < input[i].size(); j++)
    {
      input[i][j] = Activate(input[i][j]);
    }
  }
  return input;
}

std::vector<std::vector<float>> LerpEndToEnd::Derivative(std::vector<std::vector<float>> input)
{
  for (int i = 0; i < input.size(); i++)
  {
    for (int j = 0; j < input[i].size(); j++)
    {
      input[i][j] = Derivative(input[i][j]);
    }
  }
  return input;
}

void LerpEndToEnd::Backward(float z, float fg, float lr, float clamp_min, float clamp_max)
{
  for(int i = 0; i < P.size(); i++)
  {
    float dP = dlerpn_dP_i(sigmoid(z), P, i)*fg*lr;
    P[i] -= Common::Clamp(dP, clamp_min, clamp_max);
  }
}

void LerpEndToEnd::Backward(std::vector<float> z, std::vector<float> fg, float lr, float clamp_min, float clamp_max)
{
  for (int j = 0; j < z.size(); j++)
  {
    Backward(z[j], fg[j], lr, clamp_min, clamp_max);
  }
}

void LerpEndToEnd::Backward(std::vector<std::vector<float>> z, std::vector<std::vector<float>> fg, float lr, float clamp_min, float clamp_max)
{
  for (int k = 0; k < z.size(); k++)
  {
    Backward(z[k], fg[k], lr, clamp_min, clamp_max);
  }
}

bool LerpEndToEnd::IsTrainable()
{
  return true;
}

std::vector<float> LerpEndToEnd::GetControlPoints()
{
  return P;
}

void LerpEndToEnd::SetControlPoints(std::vector<float> P)
{
  this->P = P;
}

void LerpEndToEnd::SetResolution(int res, float min, float max)
{
  int old_size = P.size();
  P.resize(res);
  for (int i = old_size; i < P.size(); i++)
  {
    P[i] = Common::RandRange(min, max);
  }
}

void LerpEndToEnd::Randomize()
{
  for (int i = 0; i < P.size(); i++)
  {
    P[i] = Common::RandRange(min, max);
  }
}
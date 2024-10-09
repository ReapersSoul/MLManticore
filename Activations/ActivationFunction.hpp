#pragma once
#include <Common.hpp>
#include <vector>

class ActivationFunction
{
public:
  virtual ~ActivationFunction() = default;
  virtual float Activate(float input) = 0;
  virtual float Derivative(float input) = 0;
  virtual std::vector<float> Activate(std::vector<float> input) = 0;
  virtual std::vector<float> Derivative(std::vector<float> input) = 0;
  virtual std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> input)
  {
    throw std::runtime_error("You must implement the Activate method for matrices");
  }
  virtual std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> input)
  {
    throw std::runtime_error("You must implement the Derivative method for matrices");
  }
  virtual std::vector<std::vector<std::vector<float>>> Activate(std::vector<std::vector<std::vector<float>>> input)
  {
    throw std::runtime_error("You must implement the Activate method for tensor3s");
  }
  virtual std::vector<std::vector<std::vector<float>>> Derivative(std::vector<std::vector<std::vector<float>>> input)
  {
    throw std::runtime_error("You must implement the Derivative method for tensor3s");
  }
  virtual void Backward(float z, float fg, float lr, float clamp_min,float clamp_max)
  {
    if (IsTrainable())
    {
      throw std::runtime_error("You must implement the Backward method");
    }
  };
  virtual void Backward(std::vector<float> z, std::vector<float> fg, float lr, float clamp_min,float clamp_max)
  {
    if (IsTrainable())
    {
      throw std::runtime_error("You must implement the Backward method");
    }
  };
  virtual void Backward(std::vector<std::vector<float>> z, std::vector<std::vector<float>> fg, float lr, float clamp_min,float clamp_max)
  {
    if (IsTrainable())
    {
      throw std::runtime_error("You must implement the Backward method");
    }
  };
  virtual void Backward(std::vector<std::vector<std::vector<float>>> z, std::vector<std::vector<std::vector<float>>> fg, float lr, float clamp_min,float clamp_max)
  {
    if (IsTrainable())
    {
      throw std::runtime_error("You must implement the Backward method");
    }
  };
  virtual bool IsTrainable() { return false; };
  virtual void Randomize() { throw std::runtime_error("You must implement the Randomize method"); };
};
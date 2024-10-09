#include "BasicPerceptron.hpp"

void BasicPerceptron::Init(int InSize, ActivationFunction *af, float min, float max, float min_clamp, float max_clamp)
{
	this->InSize = InSize;
	this->af = af;
	this->min_clamp = min_clamp;
	this->max_clamp = max_clamp;
	Resize(InSize, min, max);
}

void BasicPerceptron::Resize(int InSize, float min, float max)
{
	w.resize(InSize, 0);
	x.resize(InSize, 0);
	for (int i = this->InSize; i < w.size(); i++)
	{
		w[i] = Common::RandRange(min, max);
	}
	this->InSize = InSize;
}

float BasicPerceptron::Forward(std::vector<float> x)
{
	this->x = x;
	Common::Mul(w, x, x);
	Common::Sum(x, z);
	z += b;
	return af->Activate(z);
}

std::vector<float> BasicPerceptron::Backward(float fg, float lr)
{
	float dy_dz = af->Derivative(z);
	std::vector<float> dz_dw = x;
	std::vector<float> dy_dw(InSize, 0);
	Common::Mul_Scalar(dz_dw, dy_dz * fg * lr, dy_dw);

	std::vector<float> dy_dx = w;
	Common::Mul_Scalar(dy_dx, dy_dz * fg, dy_dx);

	Common::Sub(w, dy_dw, w);
	b -= dy_dz * fg * lr;
	return dy_dx;
}

void BasicPerceptron::RandomizeWeights(float min, float max)
{
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = Common::RandRange(min, max);
	}
}

void BasicPerceptron::RandomizeBias(float min, float max)
{
	b = Common::RandRange(min, max);
}

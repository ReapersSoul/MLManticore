#include "RecurrentPerceptron.hpp"

void RecurrentPerceptron::Init(int InSize, ActivationFunction *af, float min, float max, float min_clamp, float max_clamp)
{
	this->InSize = InSize;
	this->af = af;
	this->min_clamp = min_clamp;
	this->max_clamp = max_clamp;
	Resize(InSize, min, max);
}

void RecurrentPerceptron::Resize(int InSize, float min, float max)
{
	w.resize(InSize+1, 0);
	x.resize(InSize+1, 0);
	for (int i = this->InSize; i < InSize; i++)
	{
		w[i] = Common::RandRange(min, max);
	}
	this->InSize = InSize;
}

float RecurrentPerceptron::Forward(std::vector<float> x)
{
	this->x = x;
	x.push_back(last_activation);
	Common::Mul(w, x, x);
	Common::Sum(x, z);
	z += b;
	last_activation= af->Activate(z);
	return last_activation;
}

std::vector<float> RecurrentPerceptron::Backward(float fg, float lr) {
	float dy_dz = af->Derivative(z);
	std::vector<float> dz_dw = x;
	std::vector<float> dy_dw(InSize+1, 0);
	Common::Mul_Scalar(dz_dw, dy_dz * fg * lr, dy_dw);

	std::vector<float> dy_dx = w;
	Common::Mul_Scalar(dy_dx, dy_dz * fg, dy_dx);

	Common::Sub(w, dy_dw, w);
	b -= dy_dz * fg * lr;
	return dy_dx;
}

void RecurrentPerceptron::RandomizeWeights(float min, float max)
{
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = Common::RandRange(min, max);
	}
}

void RecurrentPerceptron::RandomizeBias(float min, float max)
{
	b = Common::RandRange(min, max);
}
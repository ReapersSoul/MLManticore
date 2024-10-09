#include "MambaPerceptron.hpp"

void MambaPerceptron::Init(int InSize, ActivationFunction *af, float min, float max, float min_clamp, float max_clamp)
{
	this->InSize = InSize;
	this->af = af;
	this->min_clamp = min_clamp;
	this->max_clamp = max_clamp;
	Resize(InSize, min, max);
}

void MambaPerceptron::Resize(int InSize, float min, float max)
{
	bl.Resize(pow(InSize,2),InSize,min,max);
	w.resize(InSize);
	for (int i = this->InSize; i < InSize; i++)
	{
		w[i] = Common::RandRange(min, max);
	}
	this->InSize = InSize;
}

float MambaPerceptron::Forward(std::vector<float> x)
{
	this->x=x;
	std::vector<float> gx=x;
	gx.insert(gx.end(),w.begin(),w.end());
	w=bl.Forward(gx);
	Common::Mul(w,x,x);
	Common::Sum(x,z);
	z += b;
	return af->Activate(z);
}

std::vector<float> MambaPerceptron::Backward(float fg, float lr) {
	float dy_dz = af->Derivative(z);
	std::vector<float> dz_dw = x;
	std::vector<float> dy_dw(InSize, 0);
	Common::Mul_Scalar(dz_dw, dy_dz * fg * lr, dy_dw);
	std::vector<float> dy_dgx_stripped = bl.Backward(dy_dw,lr);
	dy_dgx_stripped.resize(InSize);
	
	std::vector<float> dy_dx = w;
	Common::Mul_Scalar(dy_dx, dy_dz * fg, dy_dx);
	Common::Add(dy_dgx_stripped,dy_dx,dy_dx);

	Common::Sub(w, dy_dw, w);
	b -= dy_dz * fg * lr;
	return dy_dx;
}

void MambaPerceptron::RandomizeWeights(float min, float max)
{
	bl.RandomizeWeights(min,max);
}

void MambaPerceptron::RandomizeBias(float min, float max)
{
	bl.RandomizeBias(min,max);
	b = Common::RandRange(min, max);
}

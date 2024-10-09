#pragma once
#include <Activations/ActivationFunction.hpp>

class Bezier : public ActivationFunction
{

	float factorial(int n);

	float stirlingsApproximation(int n);

	float binomialCoefficient(int n, int k);

	// scalar
	float B(float t, std::vector<float> P);

	float B_Prime_t(float t, std::vector<float> P);

	float B_Prime_P_i(float t, std::vector<float> P, int i, float fg, float learning_rate);

	std::vector<float> B_Prime_P(float t, std::vector<float> P, float FG, float learning_rate);

	std::vector<float> P;
	float min;
	float max;
public:
	Bezier();
	Bezier(std::vector<float> P);
	Bezier(int res, float min, float max);
	~Bezier();
	float Activate(float input);
	float Derivative(float input);
	std::vector<float> Activate(std::vector<float> input);
	std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> input);
	std::vector<float> Derivative(std::vector<float> input);
	std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> input);
	void Backward(float input, float fg, float lr, float clamp_min, float clamp_max);
	void Backward(std::vector<float> input, std::vector<float> fg, float lr, float clamp_min, float clamp_max);
	void Backward(std::vector<std::vector<float>> input, std::vector<std::vector<float>> fg, float lr, float clamp_min, float clamp_max);
	bool IsTrainable();
	std::vector<float> GetControlPoints();
	void SetControlPoints(std::vector<float> P);
	void SetResolution(int res, float min, float max);
	void Randomize();
};
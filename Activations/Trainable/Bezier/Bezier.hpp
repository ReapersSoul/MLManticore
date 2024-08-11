#pragma once
#include <Activations/ActivationFunction.hpp>

class Bezier : public ActivationFunction
{
	double factorial(int n);

	double stirlingsApproximation(int n);

	double binomialCoefficient(int n, int k);

	double B(double t, std::vector<double> P);

	double B_Prime_t(double t, std::vector<double> P);

	std::vector<double> B_Prime_t(std::vector<double> t, std::vector<double> P);

	double B_Prime_P_i(double t, std::vector<double> P, int i, double fg, double learning_rate);

	std::vector<double> B_Prime_P(std::vector<double> t, std::vector<double> P, std::vector<double> FG, double learning_rate);
	
	std::vector<double> P;
public:
	Bezier();
	Bezier(std::vector<double> P);
	Bezier(int res);
	~Bezier();
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
	void Backward(double input, double fg, double lr);
	void Backward(std::vector<double> input, std::vector<double> fg, double lr);
	bool IsTrainable();
	std::vector<double> GetControlPoints();
	void SetControlPoints(std::vector<double> P);
	void SetResolution(int res);
};
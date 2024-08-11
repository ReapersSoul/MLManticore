#pragma once
#include <Activations/ActivationFunction.hpp>

class LeakyReLU : public ActivationFunction
{
public:
	double alpha;
	LeakyReLU();
	LeakyReLU(double Alpha);
	Scalar<double> Activate(Scalar<double> input) override;
	Scalar<double> Derivative(Scalar<double> input) override;
	Vector<double> Activate(Vector<double> input) override;
	Vector<double> Derivative(Vector<double> input) override;
	Matrix<double> Activate(Matrix<double> input) override;
	Matrix<double> Derivative(Matrix<double> input) override;
	Tensor3<double> Activate(Tensor3<double> input) override;
	Tensor3<double> Derivative(Tensor3<double> input) override;
};
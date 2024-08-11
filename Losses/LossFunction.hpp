#pragma once
#include <Tensor/Tensor.hpp>
#include <Common.hpp>
#include <vector>

class LossFunction
{
public:
	virtual Scalar<double> Calculate(Scalar<double> output, Scalar<double> target) = 0;
	virtual Scalar<double> Derivative(Scalar<double> output, Scalar<double> target) = 0;
	virtual Scalar<double> Calculate(Vector<double> output, Vector<double> target) = 0;
	virtual Vector<double> Derivative(Vector<double> output, Vector<double> target) = 0;
	virtual Scalar<double> Calculate(Matrix<double> output, Matrix<double> target) = 0;
	virtual Matrix<double> Derivative(Matrix<double> output, Matrix<double> target) = 0;
	virtual Scalar<double> Calculate(Tensor3<double> output, Tensor3<double> target) = 0;
	virtual Tensor3<double> Derivative(Tensor3<double> output, Tensor3<double> target) = 0;
};
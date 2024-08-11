#pragma once
#include <Tensor/Tensor.hpp>
#include <Common.hpp>
#include <vector>

class ActivationFunction
{
public:
	virtual Scalar<double> Activate(Scalar<double> input) = 0;
	virtual Scalar<double> Derivative(Scalar<double> input) = 0;
	virtual Vector<double> Activate(Vector<double> input) = 0;
	virtual Vector<double> Derivative(Vector<double> input) = 0;
	virtual Matrix<double> Activate(Matrix<double> input) = 0;
	virtual Matrix<double> Derivative(Matrix<double> input) = 0;
	virtual Tensor3<double> Activate(Tensor3<double> input) = 0;
	virtual Tensor3<double> Derivative(Tensor3<double> input) = 0;
	virtual void Backward(Scalar<double> z,Scalar<double> fg,double lr){
		if(IsTrainable()){
			throw std::runtime_error("You must implement the Backward method");
		}
	};
	virtual void Backward(Vector<double> z,Vector<double> fg,double lr){
		if(IsTrainable()){
			throw std::runtime_error("You must implement the Backward method");
		}
	};
	virtual void Backward(Matrix<double> z,Matrix<double> fg,double lr){
		if(IsTrainable()){
			throw std::runtime_error("You must implement the Backward method");
		}
	};
	virtual void Backward(Tensor3<double> z,Tensor3<double> fg,double lr){
		if(IsTrainable()){
			throw std::runtime_error("You must implement the Backward method");
		}
	};
	virtual bool IsTrainable(){return false;};
};
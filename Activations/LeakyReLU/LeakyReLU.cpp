#include "LeakyReLU.hpp"

LeakyReLU::LeakyReLU(){
	alpha = 0.01;
}

LeakyReLU::LeakyReLU(double Alpha){
	alpha = Alpha;
}

Scalar<double> LeakyReLU::Activate(Scalar<double> x){
	return x > 0 ? x : alpha*x[0];
}

Scalar<double> LeakyReLU::Derivative(Scalar<double> x){
	return x > 0 ? 1 : alpha;
}

Vector<double> LeakyReLU::Activate(Vector<double> x){
	std::vector<double> result(x.GetSize());
	GPU_LeakyReLU(x, result, alpha);
	return result;
}

Vector<double> LeakyReLU::Derivative(Vector<double> x){
	std::vector<double> result(x.size());
	GPU_LeakyReLU_Derivative(x, result, alpha);
	return result;
}
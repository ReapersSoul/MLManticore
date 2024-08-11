#include "MambaPerceptron.hpp"

MambaPerceptron::MambaPerceptron() {
}

MambaPerceptron::~MambaPerceptron() {
}

void MambaPerceptron::Init(int size, ActivationFunction* af, double min, double max) {
	this->af = af;
	inputs=size;
	outputs=1;
	w_size=inputs*outputs;
	pl.Init(inputs+w_size,w_size,new SoftMax(),min,max);

	RandomizeGeneratorWeights(min, max);
	RandomizeBias(min, max);
}

double MambaPerceptron::Forward(std::vector<double> input) {
	ZoneScoped;
	x=input;
	double z=0;
	gx=input;
	if(w.size()!=0)
		gx.insert(gx.end(),w.begin(),w.end());
	else
		gx.insert(gx.end(),w_size,0);
	w=pl.Forward(gx);
	GPU_MulSum(x,w,z);
	return af->Activate(z+b);
}

std::vector<double> MambaPerceptron::Backward(double fg, double lr) {
	ZoneScoped;
	std::vector<double> dx(w.size());
	double dz = af->Derivative(z) * fg;
	if(af->IsTrainable())
		af->Backward(z,fg,lr);

	GPU_Mul_Scalar(w,dz,dx);
	b -= lr * dz;
	return pl.Backward(dx,lr);;
}

std::vector<double> MambaPerceptron::GetGeneratorWeights() {
	return w;
}

void MambaPerceptron::SetWeights(std::vector<double> weights) {
	w=weights;
}

void MambaPerceptron::RandomizeGeneratorWeights(double min, double max) {
	for (int i = 0; i < w.size(); i++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

void MambaPerceptron::ResizeWithRandomForNewGeneratorWeights(int size, double min, double max) {
	int wsize = w.size();
	w.resize(size);
	for (int i = wsize; i < w.size(); i++) {

		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

double MambaPerceptron::GetBias() {
	return b;
}

void MambaPerceptron::SetBias(double bias) {
	b = bias;
}

void MambaPerceptron::RandomizeBias(double min, double max) {
	double r = RandRange(min, max);
	while(r==0.0) r = RandRange(min, max);
	b = r;
}

std::vector<double> MambaPerceptron::GetX() {
	return x;
}

double MambaPerceptron::GetZ() {
	return z;
}
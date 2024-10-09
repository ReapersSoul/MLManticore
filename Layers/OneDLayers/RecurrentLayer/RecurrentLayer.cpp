#include "RecurrentLayer.hpp"

RecurrentLayer::RecurrentLayer() {
}

RecurrentLayer::~RecurrentLayer() {
}

void RecurrentLayer::Init(int InSize, int OutSize, ActivationFunction* af, float min, float max, float min_clamp, float max_clamp) {
	this->min = min;
	this->max = max;
	this->min_clamp = min_clamp;
	this->max_clamp = max_clamp;
	this->af = af;
	w.resize(InSize+OutSize);
	activationHistory.resize(OutSize);
	for (int i = 0; i < w.size(); i++) {
		w[i].resize(OutSize);
	}
	b.resize(OutSize);
	z.resize(OutSize);
	x.resize(InSize+OutSize);
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<float> RecurrentLayer::Forward(std::vector<float> input) {
	for (int i = 0; i < activationHistory.size(); i++) {
		input.push_back(activationHistory[i]);
	}	
	x=input;
	z.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		z[j] = 0.0;
		for (int i = 0; i < x.size(); i++) {
			z[j] += x[i] * w[i][j];
		}
		z[j]+=b[j];
	}
	activationHistory=af->Activate(z);
	return activationHistory;
}

std::vector<float> RecurrentLayer::Backward(std::vector<float> fg, float lr) {
	std::vector<float> dx(w.size());
	std::vector<float> dz(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		dz[j] = af->Derivative(z[j]) * fg[j];
		for (int i = 0; i < w.size(); i++) {
			dx[i]+=w[i][j] * dz[j];
			w[i][j]-=Common::Clamp(lr * dz[j] * x[i],min_clamp,max_clamp);
		}
		b[j] -= Common::Clamp(lr * dz[j],min_clamp,max_clamp);
	}

	if(af->IsTrainable()) {
		af->Backward(z,fg,lr,min_clamp,max_clamp);
	}

	return dx;
}

std::vector<std::vector<float>> RecurrentLayer::GetWeights() {
	return w;
}

void RecurrentLayer::SetWeights(std::vector<std::vector<float>> weights) {
	w=weights;
}

void RecurrentLayer::RandomizeWeights(float min, float max) {
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			float r = Common::RandRange(min, max);
			while(r==0.0) r = Common::RandRange(min, max);
			w[i][j] = r;
		}
	}
}

void RecurrentLayer::ResizeWithRandomForNewWeights(int InSize,int OutSize, float min, float max) {
	int wsize = w.size();
	w.resize(InSize+OutSize);
	for (int i = wsize; i < w.size(); i++) {
		w[i].resize(OutSize);
		for (int j = 0; j < w[i].size(); j++) {
			float r = Common::RandRange(min, max);
			while(r==0.0) r = Common::RandRange(min, max);
			w[i][j] = r;
		}
	}
}

std::vector<float> RecurrentLayer::GetBias() {
	return b;
}

void RecurrentLayer::SetBias(std::vector<float> bias) {
	b=bias;
}

void RecurrentLayer::RandomizeBias(float min, float max) {
	b.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		float r = Common::RandRange(min, max);
		while(r==0.0) r = Common::RandRange(min, max);
		b[j] = r;
	}
}

std::vector<float> RecurrentLayer::GetX() {
	return x;
}

std::vector<float> RecurrentLayer::GetZ() {
	return z;
}

std::vector<float> RecurrentLayer::GetPreviousActivation() {
	return activationHistory;
}
#include "BasicLayer.hpp"

BasicLayer::BasicLayer() {
}

BasicLayer::~BasicLayer() {
}

void BasicLayer::Init(int InSize, int OutSize, ActivationFunction* af, float min, float max,float min_clamp,float max_clamp) {
	this->min_clamp=min_clamp;
	this->max_clamp=max_clamp;
	this->af = af;
	w.resize(InSize);
	for (int i = 0; i < w.size(); i++) {
		w[i].resize(OutSize);
	}
	b.resize(OutSize);
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

void BasicLayer::Resize(int InSize, int OutSize, float min, float max)
{
	w.resize(InSize);
	for (int i = 0; i < w.size(); i++) {
		w[i].resize(OutSize,Common::RandRange(min,max));
	}
	b.resize(OutSize,Common::RandRange(min,max));
}

std::vector<std::vector<float>> TransposeMatrix(std::vector<std::vector<float>> matrix) {
	std::vector<std::vector<float>> result(matrix[0].size(), std::vector<float>(matrix.size()));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			result[j][i] = matrix[i][j];
		}
	}
	return result;
}

std::vector<float> BasicLayer::Forward(std::vector<float> input) {
	
	x=input;
	std::vector<std::vector<float>> w_t=TransposeMatrix(w);
	z.resize(w_t.size());
	for (int j = 0; j < w_t.size(); j++) {
		z[j]=0;
		Common::MulSum(x,w_t[j],z[j]);
		z[j]+=b[j];
	}
	return af->Activate(z);
}

std::vector<float> BasicLayer::Backward(std::vector<float> fg, float lr) {
	
	std::vector<float> dx(w.size());
	std::vector<float> dz=af->Derivative(z);
	dw.clear();
	dw.resize(w.size());
	for (int i = 0; i < w.size(); i++) {
		dw[i].resize(w[0].size(),0);
	}
	db.clear();
	db.resize(b.size(),0);
	for (int j = 0; j < w[0].size(); j++) {
		dz[j] = af->Derivative(z[j]) * fg[j];
		for (int i = 0; i < w.size(); i++) {
			dx[i]+=w[i][j] * dz[j];
			w[i][j]-=Common::Clamp(lr * dz[j] * x[i],min_clamp,max_clamp);
			dw[i][j]+=Common::Clamp(lr * dz[j] * x[i],min_clamp,max_clamp);
		}
		b[j] -= Common::Clamp(lr * dz[j],min_clamp,max_clamp);
		db[j] += Common::Clamp(lr * dz[j],min_clamp,max_clamp);
	}

	if(af->IsTrainable()) {
		af->Backward(z, fg, lr, -1, 1);
	}

	return dx;
}

std::vector<float> BasicLayer::GetWeights() {
	std::vector<float> weights;
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			weights.push_back(w[i][j]);
		}
	}
	return weights;
}

void BasicLayer::SetWeights(std::vector<std::vector<float>> weights) {
	w=weights;
}

void BasicLayer::RandomizeWeights(float min, float max) {
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			float r = Common::RandRange(min, max);
			while(r==0.0) r = Common::RandRange(min, max);
			w[i][j] = r;
		}
	}
}

void BasicLayer::ResizeWithRandomForNewWeights(int InSize,int OutSize, float min, float max) {
	int wsize = w.size();
	w.resize(InSize);
	for (int i = wsize; i < w.size(); i++) {
		w[i].resize(OutSize);
		for (int j = 0; j < w[i].size(); j++) {
			float r = Common::RandRange(min, max);
			while(r==0.0) r = Common::RandRange(min, max);
			w[i][j] = r;
		}
	}
}

std::vector<float> BasicLayer::GetBias() {
	return b;
}

void BasicLayer::SetBias(std::vector<float> bias) {
	b=bias;
}

void BasicLayer::RandomizeBias(float min, float max) {
	b.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		float r = Common::RandRange(min, max);
		while(r==0.0) r = Common::RandRange(min, max);
		b[j] = r;
	}
}

std::vector<std::vector<float>> BasicLayer::GetDWeights()
{
	return dw;
}

std::vector<float> BasicLayer::GetDBias()
{
	return db;
}

#include "MambaLayer.hpp"

MambaLayer::MambaLayer() {
}

MambaLayer::~MambaLayer() {
}

std::vector<std::vector<float>> MambaLayer::Split(std::vector<float> vec, int width) {
	std::vector<std::vector<float>> result;
	for (int i = 0; i < vec.size(); i+=width) {
		std::vector<float> tmp;
		for (int j = 0; j < width; j++) {
			tmp.push_back(vec[i+j]);
		}
		result.push_back(tmp);
	}
	return result;
}

std::vector<float> MambaLayer::Flatten(std::vector<std::vector<float>> vec) {
	std::vector<float> result;
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			result.push_back(vec[i][j]);
		}
	}
	return result;
}

void MambaLayer::Init(int InSize, int OutSize, ActivationFunction* af, float min, float max) {
	this->af = af;
	inputs=InSize;
	outputs=OutSize;
	w_size=InSize*OutSize;
	bl.Init(inputs+w_size,w_size,new SoftMax(),min,max);
	RandomizeGeneratorWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<float> MambaLayer::Forward(std::vector<float> input) {
	x=input;
	gx=input;
	std::vector<float> w_flat=Flatten(w);
	if(w_flat.size()!=0)
		gx.insert(gx.end(),w_flat.begin(),w_flat.end());
	else
		gx.insert(gx.end(),w_size,0);
	w=Split(bl.Forward(gx),inputs);
	z.resize(w.size());
	for (int j = 0; j < w.size(); j++) {
		z[j]=0;
		Common::MulSum(x,w[j],z[j]);
		z[j]+=b[j];
	}
	return af->Activate(z);
}

std::vector<float> MambaLayer::Backward(std::vector<float> fg, float lr) {
	std::vector<float> dx(w_size);
	std::vector<float> dz(outputs);
	Common::Mul(af->Derivative(z),fg,dz);
	if(af->IsTrainable())
		af->Backward(z,fg,lr,-1,1);

	std::vector<std::vector<float>> w_t(outputs,std::vector<float>(inputs));
	Common::TransposeMatrix(w,w_t);

	for (int j = 0; j < outputs; j++) {
		std::vector<float> tmp_dx(gx.size(),0);
		Common::Mul_Scalar(w_t[j],dz[j],tmp_dx);
		Common::Sub(dx,tmp_dx,dx);
		b[j]-=lr*dz[j];
	}
	return bl.Backward(dx,lr);
}

std::vector<std::vector<float>> MambaLayer::GetGeneratorWeights() {
	return gw;
}

void MambaLayer::SetWeights(std::vector<std::vector<float>> weights) {
	w=weights;
}

void MambaLayer::RandomizeGeneratorWeights(float min, float max) {
	for (int i = 0; i < gw.size(); i++) {
		for (int j = 0; j < gw[i].size(); j++) {
			float r = Common::RandRange(min, max);
			while(r==0.0) r = Common::RandRange(min, max);
			gw[i][j] = r;
		}
	}
}

void MambaLayer::ResizeWithRandomForNewGeneratorWeights(int InSize,int OutSize, float min, float max) {
	int old_w_size=w_size;
	inputs=InSize;
	outputs=OutSize;
	w_size=inputs*outputs;
	bl.ResizeWithRandomForNewWeights(inputs+w_size,w_size,min,max);
	std::vector<float> gw_flat=Flatten(gw);
	gw_flat.resize(w_size);
	for (int i = old_w_size; i < w_size; i++) {
		float r = Common::RandRange(min, max);
		while(r==0.0) r = Common::RandRange(min, max);
		gw_flat[i] = r;
	}
	gw=Split(gw_flat,inputs);	
}

std::vector<float> MambaLayer::GetBias() {
	return b;
}

void MambaLayer::SetBias(std::vector<float> bias) {
	b=bias;
}

void MambaLayer::RandomizeBias(float min, float max) {
	b.resize(outputs);
	for (int j = 0; j < outputs; j++) {
		float r = Common::RandRange(min, max);
		while(r==0.0) r = Common::RandRange(min, max);
		b[j] = r;
	}
}

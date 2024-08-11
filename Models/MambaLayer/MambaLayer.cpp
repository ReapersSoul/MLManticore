#include "MambaLayer.hpp"
#include <tracy/Tracy.hpp>

MambaLayer::MambaLayer() {
}

MambaLayer::~MambaLayer() {
}

std::vector<std::vector<double>> MambaLayer::Split(std::vector<double> vec, int width) {
	std::vector<std::vector<double>> result;
	for (int i = 0; i < vec.size(); i+=width) {
		std::vector<double> tmp;
		for (int j = 0; j < width; j++) {
			tmp.push_back(vec[i+j]);
		}
		result.push_back(tmp);
	}
	return result;
}

std::vector<double> MambaLayer::Flatten(std::vector<std::vector<double>> vec) {
	std::vector<double> result;
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			result.push_back(vec[i][j]);
		}
	}
	return result;
}

void MambaLayer::Init(int InSize, int OutSize, ActivationFunction* af, double min, double max) {
	this->af = af;
	inputs=InSize;
	outputs=OutSize;
	w_size=InSize*OutSize;
	pl.Init(inputs+w_size,w_size,new SoftMax(),min,max);
	RandomizeGeneratorWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<std::vector<double>> TransposeMatrix(std::vector<std::vector<double>> matrix);

std::vector<double> MambaLayer::Forward(std::vector<double> input) {
	ZoneScoped;
	x=input;
	gx=input;
	std::vector<double> w_flat=Flatten(w);
	if(w_flat.size()!=0)
		gx.insert(gx.end(),w_flat.begin(),w_flat.end());
	else
		gx.insert(gx.end(),w_size,0);
	w=Split(pl.Forward(gx),inputs);
	z.resize(w.size());
	for (int j = 0; j < w.size(); j++) {
		z[j]=0;
		GPU_MulSum(x,w[j],z[j]);
		z[j]+=b[j];
	}
	return af->Activate(z);
}

std::vector<double> MambaLayer::Backward(std::vector<double> fg, double lr) {
	ZoneScoped;
	std::vector<double> dx(w_size);
	std::vector<double> dz(outputs);
	GPU_Mul(af->Derivative(z),fg,dz);
	if(af->IsTrainable())
		af->Backward(z,fg,lr);

	std::vector<std::vector<double>> w_t(outputs,std::vector<double>(inputs));
	GPU_TransposeMatrix(w,w_t);

	for (int j = 0; j < outputs; j++) {
		std::vector<double> tmp_dx(gx.size(),0);
		GPU_Mul_Scalar(w_t[j],dz[j],tmp_dx);
		GPU_Sub(dx,tmp_dx,dx);
		b[j]-=lr*dz[j];
	}
	return pl.Backward(dx,lr);
}

std::vector<double> MambaLayer::GetGeneratorWeights() {
	std::vector<double> gw_flat;
	for (int i = 0; i < gw.size(); i++) {
		for (int j = 0; j < gw[i].size(); j++) {
			gw_flat.push_back(gw[i][j]);
		}
	}
	return gw_flat;
}

void MambaLayer::SetWeights(std::vector<std::vector<double>> weights) {
	w=weights;
}

void MambaLayer::RandomizeGeneratorWeights(double min, double max) {
	for (int i = 0; i < gw.size(); i++) {
		for (int j = 0; j < gw[i].size(); j++) {
			double r = RandRange(min, max);
			while(r==0.0) r = RandRange(min, max);
			gw[i][j] = r;
		}
	}
}

void MambaLayer::ResizeWithRandomForNewGeneratorWeights(int InSize,int OutSize, double min, double max) {
	int old_w_size=w_size;
	inputs=InSize;
	outputs=OutSize;
	w_size=inputs*outputs;
	pl.ResizeWithRandomForNewWeights(inputs+w_size,w_size,min,max);
	std::vector<double> gw_flat=Flatten(gw);
	gw_flat.resize(w_size);
	for (int i = old_w_size; i < w_size; i++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		gw_flat[i] = r;
	}
	gw=Split(gw_flat,inputs);	
}

std::vector<double> MambaLayer::GetBias() {
	return b;
}

void MambaLayer::SetBias(std::vector<double> bias) {
	b=bias;
}

void MambaLayer::RandomizeBias(double min, double max) {
	b.resize(outputs);
	for (int j = 0; j < outputs; j++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		b[j] = r;
	}
}

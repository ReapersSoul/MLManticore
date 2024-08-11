#pragma once
#include <vector>
#include <functional>
#include <thread>

#include <Activations/ActivationFunction.hpp>
#include <Models/Model.hpp>

class AutoEncoder: public Model
{
private:
	Model *encoder, *decoder;
	int EncodedSize;
public:
	AutoEncoder();
	AutoEncoder(int EncodedSize);
	~AutoEncoder();
	
	int GetEncodedSize();
	void SetEncodedSize(int EncodedSize);

	std::vector<double> Forward(std::vector<double> x);
	std::vector<double> Backward(std::vector<double> fg = std::vector<double>(1), double lr = 0.01);

	void RandomizeWeights(double min, double max);
	void RandomizeBias(double min, double max);

	template<typename EncoderType>
	EncoderType GetEncoder(){
		//check that type is child of Model
		if(!std::is_base_of<Model, EncoderType>::value){
			throw std::runtime_error("Type is not a child of Model");
		}
		return dynamic_cast<EncoderType>(encoder);
	}
	template<typename DecoderType>
	DecoderType GetDecoder(){
		//check that type is child of Model
		if(!std::is_base_of<Model, DecoderType>::value){
			throw std::runtime_error("Type is not a child of Model");
		}
		return dynamic_cast<DecoderType>(decoder);
	}

	void SetEncoder(Model* encoder);
	void SetDecoder(Model* decoder);

	void Print();
};
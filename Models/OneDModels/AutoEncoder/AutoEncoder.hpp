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

	std::vector<float> Forward(std::vector<float> x);
	std::vector<float> Backward(std::vector<float> fg = std::vector<float>(1), float lr = 0.01);

	void RandomizeWeights(float min, float max);
	void RandomizeBias(float min, float max);

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
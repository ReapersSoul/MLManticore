#include "AutoEncoder.hpp"

AutoEncoder::AutoEncoder()
{
}

AutoEncoder::AutoEncoder(int EncodedSize)
{
	this->EncodedSize = EncodedSize;
}

AutoEncoder::~AutoEncoder()
{
}

int AutoEncoder::GetEncodedSize()
{
	return EncodedSize;
}

void AutoEncoder::SetEncodedSize(int EncodedSize)
{
	this->EncodedSize = EncodedSize;
}

std::vector<float> AutoEncoder::Forward(std::vector<float> input)
{
	
}

std::vector<float> AutoEncoder::Backward(std::vector<float> fg, float lr)
{
	
}

void AutoEncoder::RandomizeWeights(float min, float max)
{
	
}

void AutoEncoder::RandomizeBias(float min, float max)
{

}

void AutoEncoder::SetEncoder(Model* encoder)
{
	this->encoder = encoder;
}

void AutoEncoder::SetDecoder(Model* decoder)
{
	this->decoder = decoder;
}

void AutoEncoder::Print()
{
	printf("AutoEncoder\n");
	printf("EncodedSize: %d\n", EncodedSize);
	printf("Encoder:\n");
	encoder->Print();
	printf("Decoder:\n");
	decoder->Print();
}
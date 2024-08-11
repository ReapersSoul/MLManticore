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

std::vector<double> AutoEncoder::Forward(std::vector<double> input)
{
	
}

std::vector<double> AutoEncoder::Backward(std::vector<double> fg, double lr)
{
	
}

void AutoEncoder::RandomizeWeights(double min, double max)
{
	
}

void AutoEncoder::RandomizeBias(double min, double max)
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
#pragma once
#include <vector>
#include <stdexcept>
#include <Activations/ActivationFunction.hpp>

class Valid2DConvolutionLayer {
	std::vector<std::vector<float>> x,k, z;
public:
	Valid2DConvolutionLayer(int KernelXSize, int KernelYSize);
	std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x, ActivationFunction *af);
	std::vector<std::vector<float>> backward(ActivationFunction *af, std::vector<std::vector<float>> ForwardGradient=std::vector<std::vector<float>>(1), float lr=.01);
	std::vector<std::vector<float>> get_x();
	std::vector<std::vector<float>> get_k();
	void set_k(std::vector<std::vector<float>> k);
	void randomize_k(float min, float max);
	void resize_with_random_for_new_k(int KernelXSize, int KernelYSize, float min, float max);
};
#include "LeakyReLU.hpp"

LeakyReLU::LeakyReLU(){
	alpha = 0.01;
}

LeakyReLU::LeakyReLU(float Alpha){
	alpha = Alpha;
}

float LeakyReLU::Activate(float x){
	return x > 0 ? x : alpha*x;
}

float LeakyReLU::Derivative(float x){
	return x > 0 ? 1 : alpha;
}

std::vector<float> LeakyReLU::Activate(std::vector<float> x){
    std::vector<float> result(x.size());
    Common::LeakyReLU(x, result, alpha);
    return result;
}

std::vector<float> LeakyReLU::Derivative(std::vector<float> x){
    std::vector<float> result(x.size());
    Common::LeakyReLU_Derivative(x, result, alpha);
    return result;
}

std::vector<std::vector<float>> LeakyReLU::Activate(std::vector<std::vector<float>> input) {
    std::vector<std::vector<float>> result(input.size());
    for (int i = 0; i < input.size(); i++) {
		Common::LeakyReLU(input[i], result[i], alpha);
	}
    return result;
}

std::vector<std::vector<float>> LeakyReLU::Derivative(std::vector<std::vector<float>> input){
    std::vector<std::vector<float>> result(input.size());
    for (int i = 0; i < input.size(); i++) {
        Common::LeakyReLU_Derivative(input[i], result[i], alpha);
    }
    return result;
}

std::vector<std::vector<std::vector<float>>> LeakyReLU::Activate(std::vector<std::vector<std::vector<float>>> input){
    std::vector<std::vector<std::vector<float>>> result(input.size());
    for (int i = 0; i < input.size(); i++) {
      result[i]= Activate(input[i]);
    }
    return result;
}

std::vector<std::vector<std::vector<float>>> LeakyReLU::Derivative(std::vector<std::vector<std::vector<float>>> input){
    std::vector<std::vector<std::vector<float>>> result(input.size());
    for (int i = 0; i < input.size(); i++) {
        result[i]= Derivative(input[i]);
    }
    return result;
}
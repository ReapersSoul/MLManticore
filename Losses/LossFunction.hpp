#pragma once
#include <Common.hpp>
#include <vector>

class LossFunction
{
public:
    virtual ~LossFunction() = default;
	virtual float Calculate(float output, float target) = 0;
	virtual float Derivative(float output, float target) = 0;
	virtual float Calculate(std::vector<float> output, std::vector<float> target) = 0;
	virtual std::vector<float> Derivative(std::vector<float> output, std::vector<float> target) = 0;
	virtual float Calculate(std::vector<std::vector<float>> output, std::vector<std::vector<float>> target){
        throw "you must implement Calculate for matrices";
    }
	virtual std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> output, std::vector<std::vector<float>> target){
        throw "you must implement Derivative for matrices";
    }
	virtual float Calculate(std::vector<std::vector<std::vector<float>>> output, std::vector<std::vector<std::vector<float>>> target){
        throw "you must implement Calculate for tensor3s";
    }
	virtual std::vector<std::vector<std::vector<float>>> Derivative(std::vector<std::vector<std::vector<float>>> output, std::vector<std::vector<std::vector<float>>> target){
        throw "you must implement Derivative for tensor3s";
    }
};
#pragma once
#include <Activations/ActivationFunction.hpp>

class LerpEndToEnd : public ActivationFunction
{

    float lerpf(float t, float min, float max)
    {
        return min + (max - min) * t;
    }

    float dlerpf_dt(float t, float min, float max){
	    return max - min;
    }

	float lerpn(float t, std::vector<float> P){
        int n = P.size()-1;
        float step = 1.0/(float)n;
        float i = floor(t/step);
        return lerpf((t-i*step)/step, P[i], P[i+1]);
    }

    float dlerpn_dt(float t, std::vector<float> P){
        int n = P.size()-1;
        float step = 1.0/(float)n;
        float i = floor(t/step);
        return dlerpf_dt((t-i*step)/step, P[i], P[i+1])/step;
    }

    float dlerpn_dP_i(float t, std::vector<float> P, int i){
        int n = P.size()-1;
        float step = 1.0/(float)n;
        float i_ = floor(t/step);
        if(i_ == i){
            return (1-(t-i*step)/step)/step;
        }else if(i_ == i+1){
            return ((t-i*step)/step)/step;
        }else{
            return 0;
        }
    }

    float sigmoid(float x){
        return 1/(1+exp(-(x)));
    }

    float dsigmoid_dx(float x){
        return (exp(-(x)))/pow(1+exp(-(x)),2);
    }

    std::vector<float> P;
	float min;
	float max;
public:
	LerpEndToEnd();
	LerpEndToEnd(std::vector<float> P);
	LerpEndToEnd(int res, float min, float max);
	~LerpEndToEnd();
	float Activate(float input);
	float Derivative(float input);
	std::vector<float> Activate(std::vector<float> input);
	std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> input);
	std::vector<float> Derivative(std::vector<float> input);
	std::vector<std::vector<float>> Derivative(std::vector<std::vector<float>> input);
	void Backward(float input, float fg, float lr, float clamp_min, float clamp_max);
	void Backward(std::vector<float> input, std::vector<float> fg, float lr, float clamp_min, float clamp_max);
	void Backward(std::vector<std::vector<float>> input, std::vector<std::vector<float>> fg, float lr, float clamp_min, float clamp_max);
	bool IsTrainable();
	std::vector<float> GetControlPoints();
	void SetControlPoints(std::vector<float> P);
	void SetResolution(int res, float min, float max);
    void Randomize();
};
#include "Bezier.hpp"

Bezier::Bezier()
{
	P = { 0, 1, 2, 3 };
}

Bezier::Bezier(std::vector<float> P)
{
	this->P = P;
}

Bezier::Bezier(int res,float min, float max)
{
  this->min = min;
  this->max = max;
  P.resize(res);
  for (int i = 0; i < P.size(); i++)
  {
    P[i] = Common::RandRange(min, max);
  }
}

Bezier::~Bezier()
{

}

float Bezier::factorial(int n)
{
	float res = 1;
	for (int i = 1; i <= n; i++)
	{
		res = res * i;
	}
	return res;
}

float Bezier::stirlingsApproximation(int n)
{
	return sqrt(2 * 3.14159265358979323846 * n) * pow(n / 2.71828182845904523536, n);
}

float Bezier::binomialCoefficient(int n, int k)
{
	return factorial(n) / (factorial(k) * factorial(n - k));
}

float Bezier::B(float t, std::vector<float> P)
{
	float b = 0;
	float n = P.size();
	for (int i = 1; i < n+1; i++)
	{
		b += binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * P[i-1];
    // if(b!=b){
    //   printf("NAN!!!!\nn: %f, i: %d, t: %f, P[i-1]: %f\n",n,i,t,P[i-1]);
    // }
	}
	return b;
}

float Bezier::B_Prime_t(float t, std::vector<float> P)
{
	float b = 0;
	float n = P.size() - 1;
	for (int i = 0; i < n; i++)
	{
		b += binomialCoefficient(n, i) * pow(1 - t, n - i - 1) * pow(t, i - 1) * (i - n * t) * P[i];
	}
	return b;
}

float Bezier::B_Prime_P_i(float t, std::vector<float> P, int i, float FG, float learning_rate)
{
	float n = P.size() - 1;
	return binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * FG * learning_rate;
}

std::vector<float> Bezier::B_Prime_P(float t, std::vector<float> P, float FG, float learning_rate)
{
	std::vector<float> P_prime(P.size());
	for (int i = 0; i < P.size(); i++)
	{
			P_prime[i] += B_Prime_P_i(t, P, i, FG, learning_rate);
	}
	return P_prime;
}

float Bezier::Activate(float input)
{
  float t=(tanh(input)+1)/2;;
	return B(t, P);
}

float Bezier::Derivative(float input)
{
  float t=(tanh(input)+1)/2;
  float dtanh=1-pow(tanh(input),2);
  float dt=dtanh/2;
  return B_Prime_t(t, P)*dt;
}

std::vector<float> Bezier::Activate(std::vector<float> input)
{
  std::vector<float> output(input.size());
  for (int i = 0; i < input.size(); i++)
  {
    output[i] = Activate(input[i]);
  }
  return output;
}

std::vector<float> Bezier::Derivative(std::vector<float> input)
{
  std::vector<float> output(input.size());
  for (int i = 0; i < input.size(); i++)
  {
    output[i] = Derivative(input[i]);
  }
  return output;
}

std::vector<std::vector<float>> Bezier::Activate(std::vector<std::vector<float>> input)
{
  std::vector<std::vector<float>> output(input.size());
  for (int i = 0; i < input.size(); i++)
  {
    output[i] = Activate(input[i]);
  }
  return output;
}

std::vector<std::vector<float>> Bezier::Derivative(std::vector<std::vector<float>> input)
{
  std::vector<std::vector<float>> output(input.size());
  for (int i = 0; i < input.size(); i++)
  {
    output[i] = Derivative(input[i]);
  }
  return output;
}

void Bezier::Backward(float z, float fg, float lr, float clamp_min,float clamp_max)
{
  z=(tanh(z)+1)/2;
	std::vector<float> P_prime = B_Prime_P({ z }, P, { fg }, lr);
	for (int i = 0; i < P.size(); i++)
	{
    float dtanh=1-pow(tanh(z),2);
    float dt=dtanh/2;
    float tmp=P[i];
    P[i] -= Common::Clamp(P_prime[i], clamp_min, clamp_max)*dt;
    if(P[i]!=P[i]){
      P[i]=tmp;
    }
	}
  P=Common::Clamp(P,min,max);
}

void Bezier::Backward(std::vector<float> z, std::vector<float> fg, float lr, float clamp_min,float clamp_max)
{
	for (int i = 0; i < z.size(); i++)
	{
    Backward(z[i], fg[i], lr, clamp_min, clamp_max);
	}
  //check if there are any nan values in P
  for(int i=0;i<P.size();i++){
    if(P[i]!=P[i]){
      printf("NAN!!!!\n");
    }
  }
}

void Bezier::Backward(std::vector<std::vector<float>> z, std::vector<std::vector<float>> fg, float lr, float clamp_min,float clamp_max)
{
  for (int i = 0; i < z.size(); i++)
  {
    Backward(z[i], fg[i], lr, clamp_min, clamp_max);
  }
}

bool Bezier::IsTrainable()
{
	return true;
}

std::vector<float> Bezier::GetControlPoints()
{
	return P;
}

void Bezier::SetControlPoints(std::vector<float> P)
{
	this->P = P;
}

void Bezier::SetResolution(int res,float min, float max)
{
	int old_size = P.size();
	P.resize(res);
	for (int i = old_size; i < P.size(); i++)
	{
		P[i] = Common::RandRange(min, max);
	}
}

void Bezier::Randomize()
{
  for(int i=0;i<P.size();i++){
    P[i]=Common::RandRange(min,max);
  }
}
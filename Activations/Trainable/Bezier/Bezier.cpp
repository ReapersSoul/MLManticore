#include "Bezier.hpp"

Bezier::Bezier()
{
	P = { 0, 1, 2, 3 };
}

Bezier::Bezier(std::vector<double> P)
{
	this->P = P;
}

Bezier::Bezier(int res)
{
	P.resize(res);
	for (int i = 0; i < P.size(); i++)
	{
		P[i] = RandRange(-.01, .01);
	}
}

Bezier::~Bezier()
{

}

double Bezier::factorial(int n)
{
	double res = 1;
	for (int i = 1; i <= n; i++)
	{
		res = res * i;
	}
	return res;
}

double Bezier::stirlingsApproximation(int n)
{
	return sqrt(2 * 3.14159265358979323846 * n) * pow(n / 2.71828182845904523536, n);
}

double Bezier::binomialCoefficient(int n, int k)
{
	return factorial(n) / (factorial(k) * factorial(n - k));
}

double Bezier::B(double t, std::vector<double> P)
{
	double b = 0;
	double n = P.size() - 1;
	for (int i = 0; i < n; i++)
	{
		b += binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * P[i];
	}
	return b;
}

double Bezier::B_Prime_t(double t, std::vector<double> P)
{
	double b = 0;
	double n = P.size() - 1;
	for (int i = 0; i < n; i++)
	{
		b += binomialCoefficient(n, i) * pow(1 - t, n - i - 1) * pow(t, i - 1) * (i - n * t) * P[i];
	}
	return b;
}

std::vector<double> Bezier::B_Prime_t(std::vector<double> t, std::vector<double> P)
{
	std::vector<double> t_prime(t.size());
	for (int i = 0; i < t.size(); i++)
	{
		t_prime[i] = B_Prime_t(t[i], P);
	}
	return t_prime;
}

double Bezier::B_Prime_P_i(double t, std::vector<double> P, int i, double FG, double learning_rate)
{
	double n = P.size() - 1;
	return binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * FG * learning_rate;
}

std::vector<double> Bezier::B_Prime_P(std::vector<double> t, std::vector<double> P, std::vector<double> FG, double learning_rate)
{
	std::vector<double> P_prime(P.size());
	for (int i = 0; i < P.size(); i++)
	{
		for (int j = 0; j < t.size(); j++)
		{
			P_prime[i] += B_Prime_P_i(t[j], P, i, FG[j], learning_rate);
		}
	}
	return P_prime;
}

double Bezier::Activate(double input)
{
	return B(input, { 0, 1, 2, 3 });
}

double Bezier::Derivative(double input)
{
	return 1;
}

std::vector<double> Bezier::Activate(std::vector<double> input)
{
	std::vector<double> result(input.size());
	for (int i = 0; i < input.size(); i++)
	{
		result[i] = input[i];
	}
	return result;
}

std::vector<double> Bezier::Derivative(std::vector<double> input)
{
	return B_Prime_t(input, P);
}

void Bezier::Backward(double z, double fg, double lr)
{
	std::vector<double> P_prime = B_Prime_P({ z }, P, { fg }, lr);
	for (int i = 0; i < P.size(); i++)
	{
		P[i] -= P_prime[i];
	}
}

void Bezier::Backward(std::vector<double> z, std::vector<double> fg, double lr)
{
	std::vector<double> P_prime = B_Prime_P(z, P, fg, lr);
	for (int i = 0; i < P.size(); i++)
	{
		P[i] -= P_prime[i];
	}
}

bool Bezier::IsTrainable()
{
	return true;
}

std::vector<double> Bezier::GetControlPoints()
{
	return P;
}

void Bezier::SetControlPoints(std::vector<double> P)
{
	this->P = P;
}

void Bezier::SetResolution(int res)
{
	int old_size = P.size();
	P.resize(res);
	for (int i = old_size; i < P.size(); i++)
	{
		P[i] = RandRange(-1, 1);
	}
}
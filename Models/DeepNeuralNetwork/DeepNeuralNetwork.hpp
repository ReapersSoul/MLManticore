#pragma once
#include <vector>
#include <functional>
#include <thread>

#include <Common.hpp>
#include <Models/Model.hpp>

class DeepNeuralNetwork : public Model
{
private:
	std::vector<Matrix<double>> w;
	std::vector<Vector<double>> b, z, x, dz;
	Vector<double> fg, dx, y;

public:
	DeepNeuralNetwork();
	~DeepNeuralNetwork();

	Vector<double> ForwardLayer(Vector<double> _x, Matrix<double> _w, Vector<double> _b, Vector<double> &_z, ActivationFunction *_af);

	Vector<double> BackwardLayer(Vector<double> _x, Matrix<double> &_w, Vector<double> &_b, Vector<double> _z, ActivationFunction *_af, Vector<double> _fg, double _lr);

	void Forward()override;

	void Backward()override;

	void SetInputVector(Vector<double> input)override;

	Vector<double> GetOutputVector()override;

	void SetForwardGradientVector(Vector<double> fg)override;

	Vector<double> GetDeltaXVector()override;

	void RandomizeWeights(double min, double max)override;

	void RandomizeBias(double min, double max)override;

	void Print()override;
};
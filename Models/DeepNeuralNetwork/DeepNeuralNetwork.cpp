#include "DeepNeuralNetwork.hpp"

DeepNeuralNetwork::DeepNeuralNetwork()
{
}

DeepNeuralNetwork::~DeepNeuralNetwork()
{
}

Vector<double> DeepNeuralNetwork::ForwardLayer(Vector<double> _x, Matrix<double> _w, Vector<double> _b, Vector<double> &_z, ActivationFunction *_af)
{
	_z.resize(_w.Rows());
	for (int j = 0; j < _w.Cols(); j++){
		_z.at(j) = 0;
		for (int i = 0; i < _x.GetSize(); i++) {
			_z.at(j) += _x.at(i) * _w.at({j,i});
		}
		_z.at(j)+=_b.at(j);
	}
	return _af->Activate(_z);
}

Vector<double> DeepNeuralNetwork::BackwardLayer(Vector<double> _x, Matrix<double> &_w, Vector<double> &_b, Vector<double> _z, ActivationFunction *_af, Vector<double> _fg, double _lr)
{
	dx.resize(_w.Rows());
	dx.Fill(0);
	//previous cpu implementation
	for (int j = 0; j < _w.Cols(); j++)
	{
		dz.at(j) = _af->Derivative(_z[j]) * _fg[j];
		for (int i = 0; i < _w.Rows(); i++)
		{
			dx[i] += _w[i][j] * dz[j];
			_w[i][j] -= _lr * dz[j] * _x[i];
		}
		_b[j] -= _lr * dz[j];
	}

	//gpu implementation
	// std::vector<std::vector<double>> W_Transpose(_w.size(), std::vector<double>(_w[0].size()));
	// GPU_TransposeMatrix(_w, W_Transpose);
	// for (int i = 0; i < W_Transpose.size(); i++)
	// {
	// 	for (int j = 0; j < W_Transpose[i].size(); j++)
	// 	{
	// 		W_Transpose[i][j] *= dz[j];
	// 	}
	// }

	return dx;
}

void DeepNeuralNetwork::Forward()
{
	for (int i = 0; i < w.size(); i++)
	{
		x[i + 1] = ForwardLayer(x[i], w[i], b[i], z[i], af);
	}
	
	return x[w.size()];
}

void DeepNeuralNetwork::Backward()
{
	std::vector<std::thread> threads;
	for (int i = w.size() - 1; i >= 0; i--)
	{
		fg = BackwardLayer(x[i], w[i], b[i], z[i], af, fg, lr);
	}
	return fg;
}

void DeepNeuralNetwork::SetInputVector(Vector<double> input)
{
	this->x[0] = input;
}

Vector<double> DeepNeuralNetwork::GetOutputVector()
{
	return y;
}

void DeepNeuralNetwork::SetForwardGradientVector(Vector<double> fg)
{
	this->fg = fg;
}

Vector<double> DeepNeuralNetwork::GetDeltaXVector()
{
	return dx;
}

void DeepNeuralNetwork::RandomizeWeights(double min, double max)
{
	for (int i = 0; i < w.size(); i++)
	{
		for (int j = 0; j < w[i].size(); j++)
		{
			for (int k = 0; k < w[i][j].size(); k++)
			{
				double r = RandRange(min, max);
				while (r == 0.0)
					r = RandRange(min, max);
				w[i][j][k] = r;
			}
		}
	}
}

void DeepNeuralNetwork::RandomizeBias(double min, double max)
{
	for (int i = 0; i < b.size(); i++)
	{
		b[i].FillRandom(min, max);
	}
}
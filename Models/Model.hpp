#pragma once
#include <vector>
#include "Tensor/Tensor.hpp"
#include "../Activations/ActivationFunction.hpp"

class Model
{
public:
	enum Type{
		T_Scalar,
		T_Vector,
		T_Matrix,
		T_Tensor
	};
protected:
	Type InputType, OutputType;
	ActivationFunction *af;
public:

	Type GetInputType(){
		return InputType;
	}

	Type GetOutputType(){
		return OutputType;
	}

	void SetActivationFunction(ActivationFunction *af){
		this->af = af;
	}

	ActivationFunction* GetActivationFunction(){
		return af;
	}

	virtual void Initialize(std::vector<Scalar<double>::IndexType> LayerSizes, Type InputType, Type OutputType, ActivationFunction *af){
		throw std::runtime_error("You must implement the Initialize method");
	}

	virtual void Initialize(std::vector<Matrix<double>::IndexType> LayerSizes, Type InputType, Type OutputType, ActivationFunction *af){
		throw std::runtime_error("You must implement the Initialize method");
	}

	virtual void Initialize(std::vector<Tensor3<double>::IndexType> LayerSizes, Type InputType, Type OutputType, ActivationFunction *af){
		throw std::runtime_error("You must implement the Initialize method");
	}

	virtual void Forward(){
		throw std::runtime_error("You must implement the Forward method");
	}

	virtual void Backward(){
		throw std::runtime_error("You must implement the Backward method");
	}

	virtual void SetInputScalar(Scalar<double> input){
		if(InputType==T_Scalar){
			throw std::runtime_error("You must implement the SetInputScalar method");
		}
		else{
			throw std::runtime_error("Input type is not scalar");
		}
	}

	virtual void SetInputVector(Vector<double> input){
		if(InputType==T_Vector){
			throw std::runtime_error("You must implement the SetInputVector method");
		}
		else{
			throw std::runtime_error("Input type is not vector");
		}
	}

	virtual void SetInputMatrix(Matrix<double> input){
		if(InputType==T_Matrix){
			throw std::runtime_error("You must implement the SetInputMatrix method");
		}
		else{
			throw std::runtime_error("Input type is not matrix");
		}
	}

	virtual void SetInputTensor(Tensor3<double> input){
		if(InputType==T_Tensor){
			throw std::runtime_error("You must implement the SetInputTensor method");
		}
		else{
			throw std::runtime_error("Input type is not tensor");
		}
	}

	virtual Scalar<double> GetOutputScalar(){
		if(OutputType==T_Scalar){
			throw std::runtime_error("You must implement the GetOutputScalar method");
		}
		else{
			throw std::runtime_error("Output type is not scalar");
		}
	}

	virtual Vector<double> GetOutputVector(){
		if(OutputType==T_Vector){
			throw std::runtime_error("You must implement the GetOutputVector method");
		}
		else{
			throw std::runtime_error("Output type is not vector");
		}
	}

	virtual Matrix<double> GetOutputMatrix(){
		if(OutputType==T_Matrix){
			throw std::runtime_error("You must implement the GetOutputMatrix method");
		}
		else{
			throw std::runtime_error("Output type is not matrix");
		}
	}

	virtual Tensor3<double> GetOutputTensor(){
		if(OutputType==T_Tensor){
			throw std::runtime_error("You must implement the GetOutputTensor method");
		}
		else{
			throw std::runtime_error("Output type is not tensor");
		}
	}

	virtual void SetForwardGradientScalar(double gradient){
		if(OutputType==T_Scalar){
			throw std::runtime_error("You must implement the SetForwardGradientScalar method");
		}
		else{
			throw std::runtime_error("Output type is not scalar");
		}
	}

	virtual void SetForwardGradientVector(Vector<double> gradient){
		if(OutputType==T_Vector){
			throw std::runtime_error("You must implement the SetForwardGradientVector method");
		}
		else{
			throw std::runtime_error("Output type is not vector");
		}
	}

	virtual void SetForwardGradientMatrix(Matrix<double> gradient){
		if(OutputType==T_Matrix){
			throw std::runtime_error("You must implement the SetForwardGradientMatrix method");
		}
		else{
			throw std::runtime_error("Output type is not matrix");
		}
	}

	virtual void SetForwardGradientTensor(Tensor3<double> gradient){
		if(OutputType==T_Tensor){
			throw std::runtime_error("You must implement the SetForwardGradientTensor method");
		}
		else{
			throw std::runtime_error("Output type is not tensor");
		}
	}

	virtual double GetDeltaXScalar(){
		if(InputType==T_Scalar){
			throw std::runtime_error("You must implement the GetDeltaXScalar method");
		}
		else{
			throw std::runtime_error("Input type is not scalar");
		}
	}

	virtual Vector<double> GetDeltaXVector(){
		if(InputType==T_Vector){
			throw std::runtime_error("You must implement the GetDeltaXVector method");
		}
		else{
			throw std::runtime_error("Input type is not vector");
		}
	}

	virtual Matrix<double> GetDeltaXMatrix(){
		if(InputType==T_Matrix){
			throw std::runtime_error("You must implement the GetDeltaXMatrix method");
		}
		else{
			throw std::runtime_error("Input type is not matrix");
		}
	}

	virtual Tensor3<double> GetDeltaXTensor(){
		if(InputType==T_Tensor){
			throw std::runtime_error("You must implement the GetDeltaXTensor method");
		}
		else{
			throw std::runtime_error("Input type is not tensor");
		}
	}
	
	virtual double GetWeightScalar(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetWeightScalar(double weight){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<double> GetWeightVector(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetWeightVector(std::vector<double> weights){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<std::vector<double>> GetWeightMatrix(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetWeightMatrix(std::vector<std::vector<double>> weights){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<std::vector<std::vector<double>>> GetWeightTensor(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetWeightTensor(std::vector<std::vector<std::vector<double>>> weights){
		throw std::runtime_error("Not implemented");
	}
	
	virtual double GetBiasScalar(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetBiasScalar(double bias){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<double> GetBiasVector(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetBiasVector(std::vector<double> bias){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<std::vector<double>> GetBiasMatrix(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetBiasMatrix(std::vector<std::vector<double>> bias){
		throw std::runtime_error("Not implemented");
	}

	virtual std::vector<std::vector<std::vector<double>>> GetBiasTensor(){
		throw std::runtime_error("Not implemented");
	}

	virtual void SetBiasTensor(std::vector<std::vector<std::vector<double>>> bias){
		throw std::runtime_error("Not implemented");
	}

	virtual void RandomizeWeights(double min, double max) = 0;
	virtual void RandomizeBias(double min, double max) = 0;
	virtual void Print() = 0;

	virtual std::vector<double> GetGeneratorWeights(){
		throw std::runtime_error("Not implemented");
	}
	
	virtual void SetGeneratorWeights(std::vector<double> weights){
		throw std::runtime_error("Not implemented");
	}

	virtual void RandomizeGeneratorWeights(double min, double max){
		throw std::runtime_error("Not implemented");
	}
};

static std::vector<double> flatten(std::vector<std::vector<double>> vec)
{
	std::vector<double> flat;
	for (int i = 0; i < vec.size(); i++)
	{
		for (int j = 0; j < vec[i].size(); j++)
		{
			flat.push_back(vec[i][j]);
		}
	}
	return flat;
}
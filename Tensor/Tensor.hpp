#pragma once
#include <vector>
#include <stdexcept>
#include <cstring>
#include <random>
#include <array>

template <typename precision, int rank>
//itterable tensor class
class Tensor
{
public:
	//if rank is 0 then index type is int, otherwise it is an array of integers
	typedef typename std::array<int,rank> IndexType;
protected:
	IndexType shape;
	int size;
	precision* data;

	template <typename other_precision>
	static other_precision RandRange(other_precision min, other_precision max) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_real_distribution<precision> dis(min, max);
		return dis(gen);
	}
public:

	//calculate position in data array from index
	static int CalculatePosition(int index,int size) {
		return index;
	}

	static int CalculatePosition(std::array<int,rank> index, std::array<int,rank> shape) {
		int position = 0;
		int multiplier = 1;
		for (int i = rank - 1; i >= 0; i--) {
			position += index[i] * multiplier;
			multiplier *= shape[i];
		}
		return position;
	}

	//calculate size of tensor from shape
	static int CalculateSize(int size) {
		return size;
	}

	static int CalculateSize(std::array<int,rank> shape) {
		int size = 1;
		for (int i = 0; i < rank; i++) {
			size *= shape[i];
		}
		return size;
	}

	static Tensor<precision, rank> FromVector(std::vector<precision> vector, IndexType shape) {
		Tensor<precision, rank> Tensor(shape);
		if (vector.size() != Tensor.GetSize()) {
			throw std::invalid_argument("Vector must have the same size as the Tensor.");
		}
		for (int i = 0; i < Tensor.GetSize(); i++) {
			Tensor.GetData()[i] = vector[i];
		}
		return Tensor;
	}

	static Tensor<precision, rank> FromArray(precision* array, IndexType shape) {
		Tensor<precision, rank> Tensor(shape);
		for (int i = 0; i < Tensor.GetSize(); i++) {
			Tensor.GetData()[i] = array[i];
		}
		return Tensor;
	}

	static Tensor<precision, rank> FromScalar(precision scalar, IndexType shape) {
		Tensor<precision, rank> Tensor(shape);
		for (int i = 0; i < Tensor.GetSize(); i++) {
			Tensor.GetData()[i] = scalar;
		}
		return Tensor;
	}

	static Tensor<precision, rank> FromRandom(IndexType shape, precision min, precision max) {
		Tensor<precision, rank> Tensor(shape);
		Tensor.FillRandom(min, max);
		return Tensor;
	}

	static Tensor<precision, rank> FromRandomNormal(IndexType shape, precision mean, precision stddev) {
		Tensor<precision, rank> Tensor(shape);
		Tensor.FillRandomNormal(mean, stddev);
		return Tensor;
	}

	static Tensor<precision, rank> FromZeros(IndexType shape) {
		Tensor<precision, rank> Tensor(shape);
		Tensor.Fill(0);
		return Tensor;
	}

	static Tensor<precision, rank> FromIndex(IndexType shape) {
		Tensor<precision, rank> Tensor(shape);
		for (int i = 0; i < Tensor.GetSize(); i++) {
			Tensor.GetData()[i] = i;
		}
		return Tensor;
	}

	Tensor() {
		this->size = 0;
		this->data = nullptr;
	}

	Tensor(IndexType shape, precision* data) {
		memcpy(this->shape, shape, rank * sizeof(int));
		this->size = 1;
		for (int i = 0; i < rank; i++) {
			this->size *= shape[i];
		}
		this->data = new precision[this->size];
		memcpy(this->data, data, this->size * sizeof(precision));
	}

	Tensor(IndexType shape, precision value) {
		memcpy(this->shape, shape, rank * sizeof(int));
		this->size = 1;
		for (int i = 0; i < rank; i++) {
			this->size *= shape[i];
		}
		this->data = new precision[this->size];
		for (int i = 0; i < this->size; i++) {
			this->data[i] = value;
		}
	}

	Tensor(IndexType shape, precision min, precision max) {
		memcpy(this->shape, shape, rank * sizeof(int));
		this->size = 1;
		for (int i = 0; i < rank; i++) {
			this->size *= shape[i];
		}
		this->data = new precision[this->size];
		for (int i = 0; i < this->size; i++) {
			this->data[i] = RandRange<precision>(min, max);
		}
	}

	Tensor(IndexType shape) {
		memcpy(this->shape, (void*)shape, rank * sizeof(int));
		this->size = 1;
		for (int i = 0; i < rank; i++) {
			this->size *= shape[i];
		}
		this->data = new precision[this->size];
	}

	Tensor(int size){
		if (rank != 1) {
			throw std::invalid_argument("Tensor must be a vector to be initialized with a single integer.");
		}
		this->size = size;
		this->shape[0] = size;
		this->data = new precision[this->size];
	}

	Tensor(int size, precision value) {
		if (rank != 1) {
			throw std::invalid_argument("Tensor must be a vector to be initialized with a single integer.");
		}
		this->size = size;
		this->shape[0] = size;
		this->data = new precision[this->size];
		for (int i = 0; i < this->size; i++) {
			this->data[i] = value;
		}
	}

	Tensor(int rows, int cols) {
		if (rank != 2) {
			throw std::invalid_argument("Tensor must be a matrix to be initialized with two integers.");
		}
		this->size = rows * cols;
		this->shape[0] = rows;
		this->shape[1] = cols;
		this->data = new precision[this->size];
	}

	Tensor(int rows, int cols, precision value) {
		if (rank != 2) {
			throw std::invalid_argument("Tensor must be a matrix to be initialized with two integers.");
		}
		this->size = rows * cols;
		this->shape[0] = rows;
		this->shape[1] = cols;
		this->data = new precision[this->size];
		for (int i = 0; i < this->size; i++) {
			this->data[i] = value;
		}
	}

	~Tensor() {
		delete[] this->data;
	}

	int GetSize() {
		return this->size;
	}

	int GetRank() {
		return rank;
	}

	IndexType GetShape() {
		return this->shape;
	}

	precision* GetData() {
		return this->data;
	}

	void push_back(precision value){
		//check rank
		if (rank != 1) {
			throw std::invalid_argument("Tensor must be a vector to push back a value.");
		}
		//resize data array
		precision* newData = new precision[this->size + 1];
		memcpy(newData, this->data, this->size * sizeof(precision));
		newData[this->size] = value;
		delete[] this->data;
		this->data = newData;
		this->size++;
	}

	void resize(int size) {
		//if rank is not vector then throw exception
		if (rank != 1) {
			throw std::invalid_argument("Tensor must be a vector to be resized with a single integer.");
		}
		if (size != this->size) {
			delete[] this->data;
			this->size = size;
			this->data = new precision[this->size];
		}
	}

	void resize(std::array<int,rank> shape) {
		int newSize = CalculateSize(shape);
		if (newSize != this->size) {
			delete[] this->data;
			this->size = newSize;
			this->data = new precision[this->size];
		}
		memcpy(this->shape.data(), shape.data(), std::max(1,rank) * sizeof(int));
	}

	void resize(IndexType shape, precision value) {
		int newSize = CalculateSize(shape);
		if (newSize != this->size) {
			delete[] this->data;
			this->size = newSize;
			this->data = new precision[this->size];
		}
		memcpy(this->shape, shape, rank * sizeof(int));
		for (int i = 0; i < this->size; i++) {
			this->data[i] = value;
		}
	}

	void resize(IndexType shape, precision min, precision max) {
		int newSize = CalculateSize(shape);
		if (newSize != this->size) {
			delete[] this->data;
			this->size = newSize;
			this->data = new precision[this->size];
		}
		memcpy(this->shape, shape, rank * sizeof(int));
		for (int i = 0; i < this->size; i++) {
			this->data[i] = RandRange<precision>(min, max);
		}
	}

	void resize(int rows, int cols) {
		//if rank is not matrix then throw exception
		if (rank != 2) {
			throw std::invalid_argument("Tensor must be a matrix to be resized with two integers.");
		}
		if (rows * cols != this->size) {
			delete[] this->data;
			this->size = rows * cols;
			this->data = new precision[this->size];
		}
		this->shape[0] = rows;
		this->shape[1] = cols;
	}

	int Rows() {
		//if rank is not matrix then throw exception
		if (rank != 2) {
			throw std::invalid_argument("Tensor must be a matrix to get the number of rows.");
		}
		return this->shape[0];
	}

	int Cols() {
		//if rank is not matrix then throw exception
		if (rank != 2) {
			throw std::invalid_argument("Tensor must be a matrix to get the number of columns.");
		}
		return this->shape[1];
	}

	precision& at(IndexType index) {
		return this->data[CalculatePosition(index, this->shape)];
	}

	precision& operator[](IndexType index) {
		return this->data[CalculatePosition(index, this->shape)];
	}

	void Fill(precision value) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] = value;
		}
	}

	void FillRandom(precision min, precision max) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] = RandRange<precision>(min, max);
		}
	}

	void FillRandomNormal(precision mean, precision stddev) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<precision> dis(mean, stddev);
		for (int i = 0; i < this->size; i++) {
			this->data[i] = dis(gen);
		}
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator+(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be added.");
		}
		Tensor<other_precision, other_rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] + other.GetData()[i];
		}
		return result;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator-(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be subtracted.");
		}
		Tensor<other_precision, other_rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] - other.GetData()[i];
		}
		return result;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator*(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be multiplied.");
		}
		Tensor<other_precision, other_rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] * other.GetData()[i];
		}
		return result;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator/(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be divided.");
		}
		Tensor<other_precision, other_rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] / other.GetData()[i];
		}
		return result;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator+(other_precision scalar) {
		Tensor<other_precision, rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] + scalar;
		}
		return result;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator-(other_precision scalar) {
		Tensor<other_precision, rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] - scalar;
		}
		return result;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator*(other_precision scalar) {
		Tensor<other_precision, rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] * scalar;
		}
		return result;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator/(other_precision scalar) {
		Tensor<other_precision, rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i] / scalar;
		}
		return result;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator+=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be added.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] += other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator-=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be subtracted.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] -= other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator*=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be multiplied.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] *= other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator/=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be divided.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] /= other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator+=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] += scalar;
		}
		return *this;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator-=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] -= scalar;
		}
		return *this;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator*=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] *= scalar;
		}
		return *this;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator/=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] /= scalar;
		}
		return *this;
	}

	template <typename other_precision, int other_rank>
	bool operator==(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			return false;
		}
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] != other.GetData()[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision, int other_rank>
	bool operator!=(Tensor<other_precision, other_rank>& other) {
		return !(*this == other);
	}

	template <typename other_precision, int other_rank>
	bool operator<(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be compared.");
		}
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] >= other.GetData()[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision, int other_rank>
	bool operator<=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be compared.");
		}
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] > other.GetData()[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision, int other_rank>
	bool operator>(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be compared.");
		}
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] <= other.GetData()[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision, int other_rank>
	bool operator>=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be compared.");
		}
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] < other.GetData()[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision>
	bool operator==(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] != scalar) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision>
	bool operator!=(other_precision scalar) {
		return !(*this == scalar);
	}

	template <typename other_precision>
	bool operator<(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] >= scalar) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision>
	bool operator<=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] > scalar) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision>
	bool operator>(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] <= scalar) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision>
	bool operator>=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			if (this->data[i] < scalar) {
				return false;
			}
		}
		return true;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator=(Tensor<other_precision, other_rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be assigned.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] = other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision, int other_rank>
	Tensor<other_precision, other_rank> operator=(Tensor<other_precision, other_rank>&& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be assigned.");
		}
		for (int i = 0; i < this->size; i++) {
			this->data[i] = other.GetData()[i];
		}
		return *this;
	}

	template <typename other_precision>
	Tensor<other_precision, rank> operator=(other_precision scalar) {
		for (int i = 0; i < this->size; i++) {
			this->data[i] = scalar;
		}
		return *this;
	}

	//dot inline
	inline precision dot(Tensor<precision, rank>& other) {
		if (this->size != other.GetSize()) {
			throw std::invalid_argument("Tensors must have the same size to be dotted.");
		}
		precision result = 0;
		for (int i = 0; i < this->size; i++) {
			result += this->data[i] * other.GetData()[i];
		}
		return result;
	}

	//conversion operators
	
	//tensor to different precision and rank
	template <typename other_precision, int other_rank>
	operator Tensor<other_precision, other_rank>() {
		if (this->size != Tensor<other_precision, other_rank>::CalculateSize(this->shape)) {
			throw std::invalid_argument("Tensors must have the same size to be converted.");
		}
		Tensor<other_precision, other_rank> result(this->shape);
		for (int i = 0; i < this->size; i++) {
			result.GetData()[i] = this->data[i];
		}
		return result;
	}
};

template <typename precision>
using Scalar= Tensor<precision, 0>;
template <typename precision>
using Vector= Tensor<precision, 1>;
template <typename precision>
using Matrix= Tensor<precision, 2>;
template <typename precision>
using Tensor3= Tensor<precision, 3>;
template <typename precision>
using Tensor4= Tensor<precision, 4>;

using IScalar= Scalar<int>;
using IVector= Vector<int>;
using IMatrix= Matrix<int>;
using ITensor3= Tensor3<int>;
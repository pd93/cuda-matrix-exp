//
// Cardiff University | Computer Science
// Module:     CM3203 One Semester Project (40 Credits)
// Title:      Parallelisation of Matrix Exponentials in C++/CUDA for Quantum Control
// Date:       2016
//
// Author:     Peter Davison
// Supervisor: Dr. Frank C Langbein
// Moderator:  Dr. Irena Spasic
//

// Precompiler include check
#ifndef matrix_h
#define matrix_h
// Include C/C++ stuff
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <random>
#include <iomanip>
#include <math.h>
// Include CUDA Stuff
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\complex.h>

class Matrix {
public:
	// STRUCTURES
	struct params {
		int scale;
		int mVal;
		thrust::host_vector<Matrix> powers;
	};
protected:
	// VARIABLES
	thrust::host_vector<thrust::complex<double>> matrix;
	int numRows, numCols;
	bool initialised;
private:
	// INTERNAL MATRIX OPERATIONS
	static Matrix& taylorMExp(Matrix& A, int k);
	static Matrix& padeMExp(Matrix& A);
	static int ell(Matrix& A, double coef, int m);
	static params getPadeParams(Matrix& A);
	static thrust::host_vector<double> getPadeCoefficients(int m);
	static Matrix& diagonalMExp(Matrix& A);
	static Matrix& zeroMExp(Matrix& A);
public:
	// CONSTRUCTORS
	Matrix();
	Matrix(int inNumRowsCols);
	Matrix(int inNumRows, int inNumCols);
	Matrix(std::vector<std::complex<double>> inMatrix, int inNumRowsCols);
	Matrix(std::vector<std::complex<double>> inMatrix, int inNumRows, int inNumCols);
	Matrix(const Matrix &obj);
	void init(int inNumRows, int inNumCols);
	// INTERNAL MATRIX OPERATIONS
	static Matrix& add(Matrix& A, Matrix& B);
	static Matrix& add(Matrix& A, double x);
	static Matrix& sub(Matrix& A, Matrix& B);
	static Matrix& sub(Matrix& A, double x);
	static Matrix& mul(Matrix& A, Matrix& B);
	static Matrix& mul(double x, Matrix& A);
	static Matrix& div(Matrix& A, Matrix& B);
	static Matrix& div(Matrix& A, double x);
	static Matrix& inv(Matrix& A);
	static Matrix& pow(Matrix& A, int x);
	static Matrix& mExp(Matrix& A, char method = ' ', int k = -1);
	// BOOLEANS
	bool isInitialised();
	bool isSquare();
	bool isDiagonal();
	bool isScalar();
	bool isIdentity();
	bool isZero();
	bool isSmall();
	// GETTERS
	thrust::complex<double> getCell(int x);
	thrust::complex<double> getCell(int row, int col);
	int getNumRows();
	int getNumCols();
	thrust::host_vector<thrust::complex<double>> Matrix::getMatrix();
	double getNorm(int n = 2);
	// SETTERS
	void setCell(int x, thrust::complex<double>);
	void setCell(int row, int col, thrust::complex<double>);
	void setNumRows(int inNumRows);
	void setNumCols(int inNumCols);
	void setMatrix(thrust::host_vector<thrust::complex<double>> inMatrix);
	void setZero();
	void setIdentity();
	void setRandom(double min, double max);
};

// GENERAL FUNCTIONS
namespace utils {
	int max(int x, int y);
	double max(double x, double y);
	int min(int x, int y);
	double min(double x, double y);
}

// OPERATOR OVERRIDES
// <<
std::ostream& operator<< (std::ostream& stream, Matrix& A);
// +
Matrix& operator+(Matrix& A, Matrix& B);
Matrix& operator+(Matrix& A, double B);
Matrix& operator+(double A, Matrix& B);
// +=
Matrix& operator+=(Matrix& A, Matrix& B);
Matrix& operator+=(Matrix& A, double B);
// -
Matrix& operator-(Matrix& A, Matrix& B);
Matrix& operator-(Matrix& A, double B);
// -=
Matrix& operator-=(Matrix& A, Matrix& B);
Matrix& operator-=(Matrix& A, double B);
// *
Matrix& operator*(Matrix& A, Matrix& B);
Matrix& operator*(Matrix& A, double B);
Matrix& operator*(double A, Matrix& B);
// *=
Matrix& operator*=(Matrix& A, Matrix& B);
Matrix& operator*=(Matrix& A, double B);
// /
Matrix& operator/(Matrix& A, Matrix& B);
Matrix& operator/(Matrix& A, double B);
// /=
Matrix& operator/=(Matrix& A, Matrix& B);
Matrix& operator/=(Matrix& A, double B);

#endif
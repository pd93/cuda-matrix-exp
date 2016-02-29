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

#ifndef matrix_h
#define matrix_h
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <random>
#include <iomanip>
#include <math.h>

class Matrix {
private:
	// Variables
	std::vector<std::complex<double>> matrix;
	int numRows, numCols;
	bool initialised;
	// Internal Matrix Functions
	static Matrix* taylorMExp(Matrix* A, int k);
	static Matrix* padeMExp(Matrix* A, int k);
	//static void PadeApproximantOfDegree(int m, Matrix* A);
	static int getPadeParams(Matrix* A);
	static std::complex<double> max(std::complex<double> x, std::complex<double> y);
	static std::complex<double> min(std::complex<double> x, std::complex<double> y);
	static double ell(Matrix* A, std::vector<std::complex<double>> coef, int m);
	static std::complex<double> oneNorm(Matrix* A);
	static std::vector<std::complex<double>> getPadeCoefficients(int m);
	static Matrix* diagonalMExp(Matrix* A);
	static Matrix* zeroMExp(Matrix* A);
public:
	// Constructors
	Matrix();
	Matrix(int inNumRowsCols);
	Matrix(int inNumRows, int inNumCols);
	Matrix(std::vector<std::complex<double>> inMatrix, int inNumRows, int inNumCols);
	Matrix(std::vector<std::complex<double>> inMatrix, int inNumRowsCols);
	Matrix(const Matrix &obj);
	void init(int inNumRows, int inNumCols);
	// Matrix Operations
	static Matrix* add(Matrix* A, Matrix* B);
	static Matrix* add(Matrix* A, double x);
	static Matrix* sub(Matrix* A, Matrix* B);
	static Matrix* sub(Matrix* A, double x);
	static Matrix* mul(Matrix* A, Matrix* B);
	static Matrix* mul(double x, Matrix* A);
	static Matrix* div(Matrix* A, Matrix* B);
	static Matrix* div(double x, Matrix* A);
	static Matrix* inv(Matrix* A);
	static Matrix* pow(Matrix* A, int x);
	static Matrix* mExp(Matrix* A, char method = ' ', int k = -1);
	// Booleans
	const bool isInitialised();
	const bool isSquare();
	const bool isDiagonal();
	const bool isScalar();
	const bool isIdentity();
	const bool isZero();
	// Getters
	const std::complex<double> getCell(int x);
	const std::complex<double> getCell(int row, int col);
	const int getNumRows();
	const int getNumCols();
	// Setters
	void setNumRows(int inNumRows);
	void setNumCols(int inNumCols);
	void setMatrix(std::vector<std::complex<double>> inMatrix);
	void setCell(int x, std::complex<double>);
	void setCell(int row, int col, std::complex<double>);
	void setZero();
	void setIdentity();
	void setRandom(double min, double max);
};

// Operators
std::ostream& operator<< (std::ostream& stream, Matrix* A);

#endif
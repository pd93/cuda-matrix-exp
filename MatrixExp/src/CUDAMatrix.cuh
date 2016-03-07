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
//#ifndef cudamatrix_h
//#define cudamatrix_h
//// Include C/C++ stuff
//#include <iostream>
//#include <string>
//#include <vector>
//#include <complex>
//#include <random>
//#include <iomanip>
//#include <math.h>
//// Include CUDA Stuff
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//class CUDAMatrix {
//public:
//	// STRUCTURES
//	struct params {
//		int scale;
//		int mVal;
//		std::vector<CUDAMatrix> powers;
//	};
//private:
//	// VARIABLES
//	std::vector<std::complex<double>> matrix;
//	int numRows, numCols;
//	bool initialised;
//	// INTERNAL MATRIX OPERATIONS
//	static CUDAMatrix& taylorMExp(CUDAMatrix& A, int k);
//	static CUDAMatrix& padeMExp(CUDAMatrix& A);
//	static int ell(CUDAMatrix& A, double coef, int m);
//	static params getPadeParams(CUDAMatrix& A);
//	static std::vector<double> getPadeCoefficients(int m);
//	static CUDAMatrix& diagonalMExp(CUDAMatrix& A);
//	static CUDAMatrix& zeroMExp(CUDAMatrix& A);
//public:
//	// CONSTRUCTORS
//	CUDAMatrix();
//	CUDAMatrix(int inNumRowsCols);
//	CUDAMatrix(int inNumRows, int inNumCols);
//	CUDAMatrix(std::vector<std::complex<double>> inMatrix, int inNumRowsCols);
//	CUDAMatrix(std::vector<std::complex<double>> inMatrix, int inNumRows, int inNumCols);
//	CUDAMatrix(const CUDAMatrix &obj);
//	void init(int inNumRows, int inNumCols);
//	// INTERNAL MATRIX OPERATIONS
//	static CUDAMatrix& add(CUDAMatrix& A, CUDAMatrix& B);
//	static CUDAMatrix& add(CUDAMatrix& A, double x);
//	static CUDAMatrix& sub(CUDAMatrix& A, CUDAMatrix& B);
//	static CUDAMatrix& sub(CUDAMatrix& A, double x);
//	static CUDAMatrix& mul(CUDAMatrix& A, CUDAMatrix& B);
//	static CUDAMatrix& mul(double x, CUDAMatrix& A);
//	static CUDAMatrix& div(CUDAMatrix& A, CUDAMatrix& B);
//	static CUDAMatrix& div(CUDAMatrix& A, double x);
//	static CUDAMatrix& inv(CUDAMatrix& A);
//	static CUDAMatrix& pow(CUDAMatrix& A, int x);
//	static CUDAMatrix& mExp(CUDAMatrix& A, char method = ' ', int k = -1);
//	// BOOLEANS
//	bool isInitialised();
//	bool isSquare();
//	bool isDiagonal();
//	bool isScalar();
//	bool isIdentity();
//	bool isZero();
//	bool isSmall();
//	// GETTERS
//	std::complex<double> getCell(int x);
//	std::complex<double> getCell(int row, int col);
//	int getNumRows();
//	int getNumCols();
//	double getNorm(int n = 2);
//	// SETTERS
//	void setCell(int x, std::complex<double>);
//	void setCell(int row, int col, std::complex<double>);
//	void setNumRows(int inNumRows);
//	void setNumCols(int inNumCols);
//	void setMatrix(std::vector<std::complex<double>> inMatrix);
//	void setZero();
//	void setIdentity();
//	void setRandom(double min, double max);
//};
//
//// GENERAL FUNCTIONS
//int max(int x, int y);
//double max(double x, double y);
//int min(int x, int y);
//double min(double x, double y);
//
//// OPERATOR OVERRIDES
//// <<
//std::ostream& operator<< (std::ostream& stream, CUDAMatrix& A);
//// +
//CUDAMatrix& operator+(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator+(CUDAMatrix& A, double B);
//CUDAMatrix& operator+(double A, CUDAMatrix& B);
//// +=
//CUDAMatrix& operator+=(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator+=(CUDAMatrix& A, double B);
//// -
//CUDAMatrix& operator-(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator-(CUDAMatrix& A, double B);
//// -=
//CUDAMatrix& operator-=(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator-=(CUDAMatrix& A, double B);
//// *
//CUDAMatrix& operator*(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator*(CUDAMatrix& A, double B);
//CUDAMatrix& operator*(double A, CUDAMatrix& B);
//// *=
//CUDAMatrix& operator*=(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator*=(CUDAMatrix& A, double B);
//// /
//CUDAMatrix& operator/(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator/(CUDAMatrix& A, double B);
//// /=
//CUDAMatrix& operator/=(CUDAMatrix& A, CUDAMatrix& B);
//CUDAMatrix& operator/=(CUDAMatrix& A, double B);
//
//#endif
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
#ifndef cudamatrix_h
#define cudamatrix_h
// Include C/C++ stuff
#include "Matrix.cuh"
// Include CUDA Stuff
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CUDAMatrix: public Matrix {
private:
	// INTERNAL MATRIX OPERATIONS
	static Matrix& taylorMExp(Matrix& A, int k);
	static Matrix& padeMExp(Matrix& A);
	static int ell(Matrix& A, double coef, int m);
	static params getPadeParams(Matrix& A);
	static Matrix& diagonalMExp(Matrix& A);
public:
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
	// BOOLEANS
	bool isInitialised();
	bool isSquare();
	bool isDiagonal();
	bool isScalar();
	bool isIdentity();
	bool isZero();
	bool isSmall();
	// GETTERS
	double getNorm(int n = 2);
	// SETTERS
	void setZero();
	void setIdentity();
	void setRandom(double min, double max);
};

#endif
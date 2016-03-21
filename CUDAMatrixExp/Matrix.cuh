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
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
// Include CUDA Stuff
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_intellisense.h"

// KERNELS
__global__ void cudaAdd(double* A, double* B, double* R, int n);
__global__ void cudaSub(double* A, double* B, double* R, int n);
__global__ void cudaMul(double* A, double* B, double* R, int n);
__global__ void cudaInv(double* A, double* R, int n, int i);
__global__ void cudaDev(double* A, double* R, int n);

class CUDAMatrix {
public:
	// STRUCTURES
	struct params {
		int scale;
		int mVal;
		std::vector<CUDAMatrix> powers;
	};
private:
	// VARIABLES
	double* h_matrix;
	double* d_matrix;
	int numRows, numCols, numEls;
	size_t size;
	bool initialised;
	// INTERNAL METHODS
	int* getPadeCoefficients(int m);
	params getPadeParams(CUDAMatrix& A);
	void alloc();
	void dealloc();
public:
	// CONSTRUCTORS & DESTRUCTOR
	CUDAMatrix();
	CUDAMatrix(int inNumRowsCols);
	CUDAMatrix(int inNumRows, int inNumCols);
	CUDAMatrix(int inNumRowsCols, std::initializer_list<double> inMatrix);
	CUDAMatrix(int inNumRows, int inNumCols, std::initializer_list<double> inMatrix);
	void init(int inNumRows, int inNumCols);
	~CUDAMatrix();
	// KERNEL CALLS
	static float CUDAMatrix::add(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::sub(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::mul(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::inv(CUDAMatrix& A, CUDAMatrix& R);
	// BOOLEANS
	bool isInitialised();
	// SYNCERS
	void syncHost();
	void syncDevice();
	// SETTERS
	void setCell(int row, int col, double val);
	void setCell(int i, double val);
	void setMatrix(int val);
	void setMatrix(const char val);
	void setMatrix(double* inMatrix);
	void setMatrix(std::initializer_list<double> inMatrix);
	// GETTERS
	int getCurRow(int i);
	int getCurCol(int i);
	double getCell(int row, int col);
	double getCell(int i);
	double* getMatrix();
	int getNumRows();
	int getNumCols();
	int getNumEls();
	size_t getSize();
};

std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A);

#endif
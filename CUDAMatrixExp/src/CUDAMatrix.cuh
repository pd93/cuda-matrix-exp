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

// CONSTANTS

#define BLOCK_SIZE 16;

// KERNELS
__global__ void cudaAdd(double* A, double* B, double* C);
__global__ void cudaSub(double* A, double* B, double* C);
__global__ void cudaMul(double* A, double* B, double* C);
__global__ void cudaInv(double* A, double* B, double* C);

class CUDAMatrix {
private:
	// VARIABLES
	double* h_matrix;
	double* d_matrix;
	int numRows, numCols, numEls;
	size_t size;
	bool initialised;
	// INTERNAL METHODS
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
	static void CUDAMatrix::mul(CUDAMatrix A, CUDAMatrix B, CUDAMatrix R);
	// BOOLEANS
	bool isInitialised();
	// SYNCERS
	void syncHost();
	void syncDevice();
	// SETTERS
	void setMatrix();
	void setMatrix(double* inMatrix);
	void CUDAMatrix::setMatrix(std::initializer_list<double> inMatrix);
	// GETTERS
	double* getMatrix();
	int getNumRows();
	int getNumCols();
	int getNumEls();
	size_t getSize();
};

std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A);

#endif
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
__global__ void cudaCholeskyInv(double* A, double* R, int ops_per_thread, int n);
__global__ void cudaCholeskyDev(double* A, double* R, int ops_per_thread, int n);
__global__ void cudaGuassJordanInv(double* A, double* R, int n, int i);
__global__ void cudaGuassJordanDev(double* A, double* R, int n);

class CUDAMatrix {
public:
	// STRUCTURES
	struct cudaParams {
		dim3 tpb; // Threads per block
		dim3 bpg; // Blocks per grid
	};
	struct padeParams {
		int scale;
		int mVal;
		std::vector<CUDAMatrix> powers;
	};
	struct cudaTimer {
		cudaEvent_t start, stop;
	};
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
	void syncHost();
	void syncDevice();
	static cudaTimer startTimer();
	static float stopTimer(cudaTimer t);
	static cudaParams getCUDAParams(int rows, int cols);
	static padeParams getPadeParams(CUDAMatrix& A);
	static int* getPadeCoefficients(int m);
public:
	// CONSTRUCTORS & DESTRUCTOR
	CUDAMatrix();
	CUDAMatrix(int inNumRowsCols);
	CUDAMatrix(int inNumRows, int inNumCols);
	CUDAMatrix(int inNumRowsCols, std::initializer_list<double> inMatrix);
	CUDAMatrix(int inNumRows, int inNumCols, std::initializer_list<double> inMatrix);
	CUDAMatrix(const CUDAMatrix &obj);
	void init(int inNumRows, int inNumCols);
	~CUDAMatrix();
	// KERNEL CALLS
	static float CUDAMatrix::add(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::sub(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::mul(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static float CUDAMatrix::tra(CUDAMatrix& A, CUDAMatrix& R);
	static float CUDAMatrix::inv(CUDAMatrix& A, CUDAMatrix& R);
	// BOOLEANS
	bool isInitialised();
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

// OPERATOR OVERRIDES
std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A);

// UTILS
namespace Utils {
	int getNumDigits(double x);
}

#endif
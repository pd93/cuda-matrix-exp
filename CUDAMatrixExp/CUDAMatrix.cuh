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
#include <random>
// Include CUDA stuff
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_intellisense.h"
#include "CUDATimer.cuh"

// KERNELS
__global__ void cudaAdd(double* A, double* B, double* R, int n);
__global__ void cudaAddScalar(double* A, double scalar, double* R, int n);
__global__ void cudaSub(double* A, double* B, double* R, int n);
__global__ void cudaSubScalar(double* A, double scalar, double* R, int n);
__global__ void cudaMul(double* A, double* B, double* R, int n);
__global__ void cudaMulScalar(double* A, double scalar, double* R, int n);
__global__ void cudaCholeskyInv(double* R, int k, int n, int stride);
__global__ void cudaCholeskyDev(double* R, int k, int n, int stride);
__global__ void cudaGuassJordanInv(double* A, double* R, int n, int i);
__global__ void cudaGuassJordanDev(double* A, double* R, int n);

class CUDAMatrix {
private:
	// STRUCTURES
	struct cudaParams {
		dim3 tpb; // Threads per block
		dim3 bpg; // Blocks per grid
	};
	struct padeParams {
		int scale;
		int mVal;
		std::vector<CUDAMatrix*> pow;
	};
	// VARIABLES
	double* h_matrix;
	double* d_matrix;
	int numRows, numCols, numEls;
	size_t size;
	bool initialised;
	// MEMORY HANDLERS
	void alloc();
	void dealloc();
	// CUDA STUFF
	void syncHost();
	void syncDevice();
	static cudaParams getCUDAParams(int rows, int cols);
	// INTERNAL PADE APPROXIMATION CODE
	static padeParams getPadeParams(CUDAMatrix& A);
	static int ell(CUDAMatrix& A, double coef, int m);
	static std::vector<double> getPadeCoefficients(int m);
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
	// MATRIX OPERATIONS
	static CUDATimer add(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static CUDATimer add(CUDAMatrix& A, double scalar, CUDAMatrix& R);
	static CUDATimer sub(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static CUDATimer sub(CUDAMatrix& A, double scalar, CUDAMatrix& R);
	static CUDATimer mul(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R);
	static CUDATimer mul(CUDAMatrix& A, double scalar, CUDAMatrix& R);
	static CUDATimer tra(CUDAMatrix& A, CUDAMatrix& R);
	static CUDATimer inv(CUDAMatrix& A, CUDAMatrix& R);			// REWRITE FOR CUDA
	static CUDATimer exp(CUDAMatrix& A, CUDAMatrix& R);
	// BOOLEANS
	bool isInitialised();
	bool isSquare();
	bool isDiagonal();
	bool isIdentity();
	bool isZero();
	bool isSmall();
	// SETTERS
	void setCell(int row, int col, double val);
	void setCell(int i, double val);
	void setMatrix(int val);
	void setMatrix(double* inMatrix);
	void setMatrix(std::initializer_list<double> inMatrix);
	void setIdentity();
	void setRandomDouble(double min = 0, double max = 1);
	void setRandomInt(int min = 0, int max = 1);
	// GETTERS
	double getNorm(int n);						// REWRITE FOR CUDA
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
namespace utils {
	int getNumDigits(double x);
	int max(int x, int y);
	double max(double x, double y);
	int min(int x, int y);
	double min(double x, double y);
}

#endif
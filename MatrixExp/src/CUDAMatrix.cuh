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
//#include <string>
//#include <complex>
//#include <random>
// Include CUDA Stuff
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CUDAMatrix {
private:
	// VARIABLES
	std::vector<double>* d_matrix;
	std::vector<double>* h_matrix;
	int numRows, numCols;
	size_t size;
	bool initialised;
public:
	// CONSTRUCTORS
	CUDAMatrix();
	CUDAMatrix(int inNumRowsCols);
	CUDAMatrix(int inNumRows, int inNumCols);
	CUDAMatrix(int inNumRowsCols, std::vector<double> inMatrix);
	CUDAMatrix(int inNumRows, int inNumCols, std::vector<double> inMatrix);
	void init(int inNumRows, int inNumCols, std::vector<double> inMatrix);
	// DESTRUCTOR
	~CUDAMatrix();
	// BOOLEANS
	bool isInitialised();
	// GETTERS
	std::vector<double> getMatrix();
	int getNumRows();
	int getNumCols();
	size_t getSize();
};

std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A);

#endif
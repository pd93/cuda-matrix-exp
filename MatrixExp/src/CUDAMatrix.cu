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

// Include header file
#include "CUDAMatrix.cuh"

// Kernels

// CONSTRUCTORS

// Default constructor. Creates an uninitialsed instance of a matrix
CUDAMatrix::CUDAMatrix() {
	initialised = false;
}
// Creates an instance of a square matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRowsCols) {
	std::vector<double> h_matrix(inNumRowsCols*inNumRowsCols);
	init(inNumRowsCols, inNumRowsCols, h_matrix);
}
// Creates an instance of an (n x m) matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols) {
	std::vector<double> h_matrix(inNumRows*inNumCols);
	init(inNumRows, inNumCols, h_matrix);
}
// Creates an instance of a square matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRowsCols, std::vector<double> inMatrix) {
	init(inNumRowsCols, inNumRowsCols, inMatrix);
}
// Creates an instance of an (n x m) matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols, std::vector<double> inMatrix) {
	init(inNumRows, inNumCols, inMatrix);
}
// Initialiser. Resizes the matrix and sets the values to 0
void CUDAMatrix::init(int inNumRows, int inNumCols, std::vector<double> inMatrix) {
	size = sizeof(double)*inNumRows*inNumCols;
	h_matrix = (std::vector<double>*) malloc(size);
	memcpy(h_matrix, &inMatrix, size);
	cudaMalloc((void**)&d_matrix, size);
	cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
	initialised = true;
}

// DESTRUCTOR

CUDAMatrix::~CUDAMatrix() {
	free(h_matrix);
	cudaFree(d_matrix);
}

// BOOLEANS

bool CUDAMatrix::isInitialised() {
	return initialised;
}

// GETTERS

std::vector<double> CUDAMatrix::getMatrix() {
	if (isInitialised()) {
		cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
		return *h_matrix;
	} else {
		throw;
	}
}

int CUDAMatrix::getNumRows() {
	return numRows;
}

int CUDAMatrix::getNumCols() {
	return numCols;
}

size_t CUDAMatrix::getSize() {
	return size;
}

// OPERATOR OVERRIDES

// <<
std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A) {
	std::vector<double> matrix(A.getMatrix());
	if (A.isInitialised()) {
		double cell;
		for (int c1 = 0; c1 < A.getNumRows(); c1++) {
			oStream << "|";
			for (int c2 = 0; c2 < A.getNumCols(); c2++) {
				cell = matrix[c1*c2+c1];
				if (abs(cell - (int) (cell) != 0)) {
					// Decimal
					oStream << " " << std::setprecision(3) << std::fixed << cell;
				} else {
					// Integer
					oStream << " " << std::setprecision(0) << std::fixed << cell;
				}
			}
			oStream << " |" << std::endl;
		}
		return oStream;
	} else {
		throw;
	}
}
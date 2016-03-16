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

// KERNELS

__global__ void cudaAdd(double* A, double* B, double* C) {

}

__global__ void cudaSub(double* A, double* B, double* C) {

}

__global__ void cudaMul(double* A, double* B, double* C) {

}

__global__ void cudaInv(double* A, double* B, double* C) {

}

// INTERNAL METHODS

// Allocate memory on the host and device
void CUDAMatrix::alloc() {
	h_matrix = (double*) malloc(size);
	cudaMalloc((void**) &d_matrix, size);
}
// Deallocate memory on the host and device
void CUDAMatrix::dealloc() {
	free(h_matrix);
	cudaFree(d_matrix);
}

// CONSTRUCTORS

// Default constructor. Creates an uninitialsed instance of a matrix
CUDAMatrix::CUDAMatrix() {
	initialised = false;
}
// Creates an instance of a square matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRowsCols) {
	init(inNumRowsCols, inNumRowsCols);
	setMatrix();
	syncDevice();
}
// Creates an instance of an (n x m) matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
	setMatrix();
	syncDevice();
}
// Creates an instance of a square matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRowsCols, std::initializer_list<double> inMatrix) {
	init(inNumRowsCols, inNumRowsCols);
	setMatrix(inMatrix);
	syncDevice();
}
// Creates an instance of an (n x m) matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols, std::initializer_list<double> inMatrix) {
	init(inNumRows, inNumCols);
	setMatrix(inMatrix);
	syncDevice();
}
// Initialiser
void CUDAMatrix::init(int inNumRows, int inNumCols) {
	numRows = inNumRows;
	numCols = inNumCols;
	numEls = inNumRows*inNumCols;
	size = sizeof(double)*numEls;
	alloc();
	initialised = true;
}
// Destructor
CUDAMatrix::~CUDAMatrix() {
	dealloc();
}

// KERNEL CALLS

void CUDAMatrix::mul(CUDAMatrix A, CUDAMatrix B, CUDAMatrix R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(A.getNumCols() / dimBlock.x, A.getNumRows() / dimBlock.y);
		cudaMul <<< dimGrid, dimBlock >>> (A.d_matrix, B.d_matrix, R.d_matrix);
	} else {
		throw;
	}
}

// BOOLEANS

bool CUDAMatrix::isInitialised() {
	return initialised;
}

// SYNCERS

void CUDAMatrix::syncHost() {
	if (isInitialised()) {
		cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
	} else {
		throw;
	}
}

void CUDAMatrix::syncDevice() {
	if (isInitialised()) {
		cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
	} else {
		throw;
	}
}

// SETTERS

void CUDAMatrix::setMatrix() {
	if (isInitialised()) {
		for (int c1 = 0; c1 < getNumEls(); c1++) {
			h_matrix[c1] = 0;
		}
		syncDevice();
	} else {
		throw;
	}
}

void CUDAMatrix::setMatrix(double* inMatrix) {
	if (isInitialised()) {
		memcpy(&h_matrix, inMatrix, size);
		syncDevice();
	} else {
		throw;
	}
}

void CUDAMatrix::setMatrix(std::initializer_list<double> inMatrix) {
	if (isInitialised()) {
		if (inMatrix.size() == getNumEls()) {
			std::copy(inMatrix.begin(), inMatrix.end(), h_matrix);
			syncDevice();
		} else {
			throw;
		}
	} else {
		throw;
	}
}

// GETTERS

double* CUDAMatrix::getMatrix() {
	if (isInitialised()) {
		syncHost();
		return h_matrix;
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

int CUDAMatrix::getNumEls() {
	return numEls;
}

size_t CUDAMatrix::getSize() {
	return size;
}

// OPERATOR OVERRIDES

// <<
std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A) {
	double* matrix(A.getMatrix());
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
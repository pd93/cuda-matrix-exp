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

// INTERNAL MATRIX OPERATIONS

// Finds the exponential of a matrix using the Taylor Series
Matrix& CUDAMatrix::taylorMExp(Matrix& A, int k) {
	return A;
}
// Finds the exponential of a matrix using the Pade Approximation
Matrix& CUDAMatrix::padeMExp(Matrix& A) {
	return A;
}
// Something to do with pade scaling                                   NEEDS AMMENDING
int CUDAMatrix::ell(Matrix& A, double coef, int m) {
	return 0;
}
// Gets the parameters needed for the pade approximation
CUDAMatrix::params CUDAMatrix::getPadeParams(Matrix& A) {
	params p;
	return p;
}
// Finds the exponential of a diagonal matrix
Matrix& CUDAMatrix::diagonalMExp(Matrix& A) {
	return A;
}

// EXTERNAL MATRIX OPERATIONS

// Adds two matrices together
Matrix& CUDAMatrix::add(Matrix& A, Matrix& B) {
	if (A.isInitialised() && B.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ar == br && ac == bc) {
			return A;
		} else {
			// Error! Cannot add these matrices
			throw (201);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Adds a scalar to a matrix
Matrix& CUDAMatrix::add(Matrix& A, double x) {
	if (A.isInitialised()) {
		return A;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Subtracts one matrix from another
Matrix& CUDAMatrix::sub(Matrix& A, Matrix& B) {
	if (A.isInitialised() && B.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ar == br && ac == bc) {
			return A;
		} else {
			// Error! Cannot subtract these matrices
			throw (201);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Subtracts a scalar from a matrix
Matrix& CUDAMatrix::sub(Matrix& A, double x) {
	if (A.isInitialised()) {
		return A;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Multiplies two matrices together
Matrix& CUDAMatrix::mul(Matrix& A, Matrix& B) {
	if (A.isInitialised() && B.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ac == br) {
			return A;
		} else {
			// Error! Cannot multiply these matrices together
			throw (203);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Multiplies a matrix by a scalar
Matrix& CUDAMatrix::mul(double x, Matrix& A) {
	if (A.isInitialised()) {
		return A;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Finds the inverse of a matrix
Matrix& CUDAMatrix::inv(Matrix& A) {
	if (A.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		if (ar == ac) {
			return A;
		} else {
			// Error! Cannot find the inverse of this matrix
			throw (205);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Finds the nth power of a matrix
Matrix& CUDAMatrix::pow(Matrix& A, int x) {
	if (A.isInitialised()) {
		return A;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Finds the exponential of a matrix

// BOOLEANS

// Check if a matrix is diagonal
bool CUDAMatrix::isDiagonal() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (c1 != c2 && getCell(c1, c2).real() != 0) {
					return false;
				}
			}
		}
		return true;
	} else {
		// Error! Cannot determine if matrix is diagonal before initialisation
		throw (106);
	}
}
// Check if a matrix is a scalar matrix
bool CUDAMatrix::isScalar() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if ((c1 != c2 && getCell(c1, c2).real() != 0) || (c1 == c2 && getCell(c1, c2) != getCell(0, 0))) {
					return false;
				}
			}
		}
		return true;
	} else {
		// Error! Cannot determine if matrix is scalar before initialisation
		throw (107);
	}
}
// Check if a matrix is an identity matrix
bool CUDAMatrix::isIdentity() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if ((c1 != c2 && getCell(c1, c2).real() != 0) || (c1 == c2 && getCell(c1, c2).real() != 1)) {
					return false;
				}
			}
		}
		return true;
	} else {
		// Error! Cannot determine if matrix is an identity matrix before initialisation
		throw (108);
	}
}
// Check if a matrix is a zero matrix
bool CUDAMatrix::isZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (getCell(c1, c2).real() != 0) {
					return false;
				}
			}
		}
		return true;
	} else {
		// Error! Cannot determine if matrix is a zero matrix before initialisation
		throw (109);
	}
}

// GETTERS

// Find the normal of a matrix
double CUDAMatrix::getNorm(int n) {
	int c1, c2;
	double sum, max = 0;
	if (n == 1) {
		// 1 Norm
		for (c1 = 0; c1 < getNumCols(); c1++) {
			sum = 0;
			for (c2 = 0; c2 < getNumRows(); c2++) {
				sum += abs(getCell(c1, c2).real());
			}
			if (sum > max) {
				max = sum;
			}
		}
		return max;
	} else if (n == INFINITY) {
		// Inf Norm
		for (c1 = 0; c1 < getNumRows(); c1++) {
			sum = 0;
			for (c2 = 0; c2 < getNumCols(); c2++) {
				sum += abs(getCell(c1, c2).real());
			}
			if (sum > max) {
				max = sum;
			}
		}
		return max;
	} else {
		// Euclidian
		sum = 0;
		for (c1 = 0; c1 < getNumCols()*getNumRows(); c1++) {
			sum += std::pow(getCell(c1).real(), n);
		}
		return std::pow(sum, 1.0 / n);
	}
}

// SETTERS

// Set the matrix to a zero matrix
void CUDAMatrix::setZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows*numCols; c1++) {
			setCell(c1, 0);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Set the matrix to an identity matrix
void CUDAMatrix::setIdentity() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (c1 == c2) {
					setCell(c1, c2, 1);
				} else {
					setCell(c1, c2, 0);
				}
			}
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Set the matrix to a random matrix between limits
void CUDAMatrix::setRandom(double min, double max) {
	if (initialised) {
		std::default_random_engine rng;
		std::uniform_int_distribution<int> gen((int) floor(min), (int) floor(max));
		for (int c1 = 0; c1 < numRows*numCols; c1++) {
			int randomI = gen(rng);
			setCell(c1, randomI);
		}
	}
}
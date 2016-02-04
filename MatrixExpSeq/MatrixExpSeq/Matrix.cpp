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

#include "Matrix.hpp"

// Constructors

Matrix::Matrix() {
	initialised = false;
}

Matrix::Matrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
}

Matrix::Matrix(std::vector<std::vector<double>> inMatrix) {
	init((int) inMatrix.size(), (int) inMatrix[0].size());
	setMatrix(inMatrix);
}

Matrix::Matrix(const Matrix &obj) {
	// COPY CONSTRUCTOR NOT COMPLETE
	matrix = obj.matrix;
	numRows = obj.numRows;
	numCols = obj.numCols;
	initialised = obj.initialised;
}

void Matrix::init(int inNumRows, int inNumCols) {
	numRows = inNumRows;
	numCols = inNumCols;
	matrix.resize(numRows);
	for (int i = 0; i < numRows; i++) {
		matrix[i].resize(numCols);
	}
	initialised = true;	
	setZero();
}

// Matrix Functions

Matrix Matrix::add(Matrix matrixB) {
	if (initialised && matrixB.initialised) {
		if (numRows == matrixB.getNumRows() && numCols == matrixB.getNumCols()) {
			Matrix matrixC(numRows, numCols);
			for (int c1 = 0; c1 < numRows; c1++) {
				for (int c2 = 0; c2 < numCols; c2++) {
					matrixC.matrix[c1][c2] = matrix[c1][c2] + matrixB.matrix[c1][c2];
				}
			}
			return matrixC;
		}
		else {
			// Error! Cannot add these matrices
			throw (201);
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix Matrix::mul(Matrix matrixB) {
	if (initialised && matrixB.initialised) {
		if (numCols == matrixB.getNumRows()) {
			Matrix matrixC(numRows, matrixB.getNumCols());
			double cell;
			for (int c1 = 0; c1 < numRows; c1++) {
				for (int c2 = 0; c2 < matrixB.getNumCols(); c2++) {
					cell = 0;
					for (int c3 = 0; c3 < matrixB.getNumRows(); c3++) {
						cell += matrix[c1][c3] * matrixB.matrix[c3][c2];
					}
					matrixC.matrix[c1][c2] = cell;
				}
			}
			return matrixC;
		}
		else {
			// Error! Cannot multiply these matrices together
			throw (202);
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix Matrix::mul(double m) {
	if (initialised) {
		Matrix matrixB(numRows, numCols);
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				matrixB.matrix[c1][c2] = matrix[c1][c2] * m;
			}
		}
		return matrixB;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix Matrix::pow(int p) {
	if (initialised) {
		Matrix matrixB(*this);
		for (int c1 = 0; c1 < p - 1; c1++) {
			matrixB = matrixB.mul(matrix);
		}
		return matrixB;
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix Matrix::taylorMExp(int k) {
	double nfact = 1;
	double coef;
	Matrix matrixAn, term;
	Matrix matrixB(numRows, numCols);
	matrixB.setIdentity();
	for (int n = 1; n <= k; n++) {
		nfact *= n;
		coef = 1.0 / nfact;
		matrixAn = pow(n);
		term = matrixAn.mul(coef);
		matrixB = matrixB.add(term);
	}
	return matrixB;
}

Matrix Matrix::padeMExp(int k) {
	double nfact = 1;
	double coef;
	Matrix matrixAn, term;
	Matrix matrixB(numRows, numCols);
	matrixB.setIdentity();
	for (int n = 1; n <= k; n++) {
		nfact *= n;
		coef = 1.0 / nfact;
		matrixAn = pow(n);
		term = matrixAn.mul(coef);
		matrixB = matrixB.add(term);
	}
	return matrixB;
}

Matrix Matrix::diagonalMExp() {
	Matrix matrixB(*this);
	for (int c1 = 0; c1 < numRows; c1++) {
		matrixB.matrix[c1][c1] = exp(matrixB.matrix[c1][c1]);
	}
	return matrixB;
}

Matrix Matrix::zeroMExp() {
	Matrix matrixB(*this);
	return matrixB;
}

Matrix Matrix::mExp(int k, char method) {
	if (initialised) {
		// Special Cases
		if (isDiagonal()) {
			return diagonalMExp();
		}
		else if (isZero()) {
			return zeroMExp();
		}
		// Ordinary Cases
		else {
			switch (method) {
				default:
					return taylorMExp(k);
				case 't':
					return taylorMExp(k);
				case 'p':
					return padeMExp(k);
			}
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

void Matrix::setZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				matrix[c1][c2] = 0;
			}
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

void Matrix::setIdentity() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (c1 == c2) {
					matrix[c1][c2] = 1;
				}
				else {
					matrix[c1][c2] = 0;
				}
			}
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

// Booleans

bool Matrix::isSquare() {
	if (initialised) {
		if (numCols == numRows) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		// Error! Cannot determine if matrix is square before initialisation
		throw (105);
	}
}

bool Matrix::isDiagonal() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (c1 != c2 && matrix[c1][c2] != 0) {
					return false;
				}
			}
		}
		return true;
	}
	else {
		// Error! Cannot determine if matrix is diagonal before initialisation
		throw (106);
	}
}

bool Matrix::isScalar() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if ((c1 != c2 && matrix[c1][c2] != 0) || (c1 == c2 && matrix[c1][c2] != matrix[0][0])) {
					return false;
				}
			}
		}
		return true;
	}
	else {
		// Error! Cannot determine if matrix is scalar before initialisation
		throw (107);
	}
}

bool Matrix::isIdentity() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if ((c1 != c2 && matrix[c1][c2] != 0) || (c1 == c2 && matrix[c1][c2] != 1)) {
					return false;
				}
			}
		}
		return true;
	}
	else {
		// Error! Cannot determine if matrix is an identity matrix before initialisation
		throw (108);
	}
}

bool Matrix::isZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (matrix[c1][c2] != 0) {
					return false;
				}
			}
		}
		return true;
	}
	else {
		// Error! Cannot determine if matrix is a zero matrix before initialisation
		throw (109);
	}
}

// Getters

int Matrix::getNumRows() {
	if (initialised) {
		return numRows;
	}
	else {
		// Error! Cannot determine number of rows in matrix before initialisation
		throw (103);
	}
}

int Matrix::getNumCols() {
	if (initialised) {
		return numCols;
	}
	else {
		// Error! Cannot determine number of columns in matrix before initialisation
		throw (104);
	}
}

// Setters

void Matrix::setMatrix(std::vector<std::vector<double>> inMatrix) {
	matrix = inMatrix;
}

void Matrix::setCell(int row, int col, double value) {
	matrix[row][col] = value;
}

// Output

void Matrix::printm(int precision) {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			printf("|");
			for (int c2 = 0; c2 < numCols; c2++) {
				if (matrix[c1][c2] >= 0) {
					if (abs(matrix[c1][c2] - (int) matrix[c1][c2]) > 0) {
						// Decimal
						printf(" %.*f", precision, matrix[c1][c2]);
					}
					else {
						// Integer
						printf(" %.0f", matrix[c1][c2]);
					}
				}
				else {
					printf(" -");
				}
			}
			printf(" |\n");
		}
	}
	else {
		// Error! Cannot print matrix before initialisation
		throw (102);
	}
}
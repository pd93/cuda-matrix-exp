//
// Author: Pete Davison (2016)
//

#include "Matrix.hpp"

// Constructors

Matrix::Matrix() {
	initialised = false;
}

Matrix::Matrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
}

Matrix::Matrix(std::vector<std::vector<int>> inMatrix) {
	init(inMatrix.size(), inMatrix[0].size());
	setMatrix(inMatrix);
}

void Matrix::init(int inNumRows, int inNumCols) {
	// Set height/width of matrix
	numRows = inNumRows;
	numCols = inNumCols;
	matrix.resize(numRows);
	for (int i = 0; i < numRows; i++) {
		matrix[i].resize(numCols);
	}
	initialised = true;	
	// Set values to zero
	setZero();
}

// Matrix Functions

Matrix Matrix::mul(Matrix matrixB) {
	if (initialised && matrixB.initialised) {
		Matrix matrixC;
		if (numCols == matrixB.getNumRows()) {
			matrixC.init(numRows, matrixB.getNumCols());
			printf(matrixC.toString().c_str());
			getchar();
			int cell = 0;
			for (int c1 = 0; c1 < numRows; c1++) {
				for (int c2 = 0; c2 < matrixB.getNumCols(); c2++) {
					for (int c3 = 0; c3 < matrixB.getNumRows(); c3++) {
						cell += matrix[c3][c1] * matrixB.matrix[c2][c3];
					}
					matrixC.matrix[c2][c1] = cell;
					cell = 0;
				}
			}
			getchar();
		}
		else {
			printf("Error! Cannot multiply these matrices together.\n");
		}
		return matrixC;
	}
	else {
		printf("Error! Cannot perform matrix operations before initialisation.\n");
	}
}

Matrix Matrix::pow(int power) {
	if (initialised) {
		Matrix matrixB;
		memcpy(&matrixB.matrix, &matrix, sizeof(matrix));
		printf("%s\n", toString().c_str());
		printf("%s\n", matrixB.toString().c_str());
		for (int c1 = 0; c1 < power - 1; c1++) {
			matrixB = mul(matrixB);
		}
		return matrixB;
	}
	else {
		printf("Error! Cannot perform matrix operations before initialisation.\n");
	}
}

Matrix Matrix::exp(int order) {
	if (initialised) {
		Matrix A(numCols, numRows);
		for (int k = 0; k <= order; k++) {

		}
		return A;
	}
	else {
		printf("Error! Cannot perform matrix operations before initialisation.\n");
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
		printf("Error! Cannot perform matrix operations before initialisation.\n");
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
		printf("Error! Cannot perform matrix operations before initialisation.\n");
	}
}

// Output

std::string Matrix::toString() {
	if (initialised) {
		std::string output("");
		for (int c1 = 0; c1 < numRows; c1++) {
			output.append("|");
			for (int c2 = 0; c2 < numCols; c2++) {
				if (matrix[c1][c2] >= 0) {
					output.append(" ");
					output.append(std::to_string(matrix[c1][c2]));
				} else {
					output.append(" -");
				}
			}
			output.append(" |\n");
		}
		return output;
	}
	else {
		printf("Error! Cannot print matrix before initialisation.\n");
		return std::string("");
	}
}

// Getters

int Matrix::getNumRows() {
	if (initialised) {
		return numRows;
	}
	else {
		printf("Error! Cannot determine number of rows in matrix before initialisation.\n");
		return 0;
	}
}

int Matrix::getNumCols() {
	if (initialised) {
		return numCols;
	}
	else {
		printf("Error! Cannot determine number of columns in matrix before initialisation.\n");
		return 0;
	}
}

// Setters

void Matrix::setMatrix(std::vector<std::vector<int>> inMatrix) {
	matrix = inMatrix;
}

void Matrix::setCell(int row, int col, int value) {
	matrix[row][col] = value;
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
		printf("Error! Cannot determine if matrix is square before initialisation.\n");
		return false;
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
		printf("Error! Cannot determine if matrix is diagonal before initialisation.\n");
		return false;
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
		printf("Error! Cannot determine if matrix is scalar before initialisation.\n");
		return false;
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
		printf("Error! Cannot determine if matrix is an identity matrix before initialisation.\n");
		return false;
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
		printf("Error! Cannot determine if matrix is a zero matrix before initialisation.\n");
		return false;
	}
}
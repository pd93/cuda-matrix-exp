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

Matrix* Matrix::add(Matrix* A, Matrix* B) {
	if (A->initialised && B->initialised) {
		int ar = A->getNumRows();
		int ac = A->getNumCols();
		int br = B->getNumRows();
		int bc = B->getNumCols();
		if (ar == br && ac == bc) {
			Matrix *R = new Matrix(ar, ac);
			for (int c1 = 0; c1 < ar; c1++) {
				for (int c2 = 0; c2 < ac; c2++) {
					R->setCell(c1, c2, A->getCell(c1, c2) + B->getCell(c1, c2));
				}
			}
			return R;
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

Matrix* Matrix::mul(Matrix* A, Matrix* B) {
	if (A->initialised && B->initialised) {
		int ar = A->getNumRows();
		int ac = A->getNumCols();
		int br = B->getNumRows();
		int bc = B->getNumCols();
		if (ac == br) {
			Matrix* R = new Matrix(ar, bc);
			double cell;
			for (int c1 = 0; c1 < ar; c1++) {
				for (int c2 = 0; c2 < bc; c2++) {
					cell = 0;
					for (int c3 = 0; c3 < br; c3++) {
						cell += A->getCell(c1, c3) * B->getCell(c3, c2);
					}
					R->setCell(c1, c2, cell);
				}
			}
			return R;
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

Matrix* Matrix::mul(Matrix* A, double B) {
	if (A->initialised) {
		int ar = A->getNumRows();
		int ac = A->getNumCols();
		Matrix* R = new Matrix(ar, ac);
		for (int c1 = 0; c1 < ar; c1++) {
			for (int c2 = 0; c2 < ac; c2++) {
				R->setCell(c1, c2, A->getCell(c1, c2) * B);
			}
		}
		return R;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix* Matrix::pow(Matrix* A, int x) {
	if (A->initialised) {
		Matrix* R = new Matrix(A->getNumRows(), A->getNumCols());
		R->setIdentity();
		for (int c1 = 1; c1 <= x; c1++) {
			R = Matrix::mul(A, R);
		}
		return R;
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix* Matrix::taylorMExp(Matrix* A, int k) {
	double nfact = 1;
	double coef;
	Matrix* An;
	Matrix* T;
	Matrix* R = new Matrix(A->getNumRows(), A->getNumCols());
	R->setIdentity();
	for (int n = 1; n <= k; n++) {
		nfact *= n;
		coef = 1.0 / nfact;
		An = Matrix::pow(A, n);
		T = Matrix::mul(An, coef);
		R = Matrix::add(R, T);
	}
	return R;
}

Matrix* Matrix::padeMExp(Matrix* A, int k) {
	double nfact = 1;
	//double coef;
	//Matrix* An;
	//Matrix* T;
	Matrix* R = new Matrix(A->getNumRows(), A->getNumCols());
	R->setIdentity();
	return R;
}

Matrix* Matrix::diagonalMExp(Matrix* A) {
	Matrix* R = new Matrix(A->getNumRows(), A->getNumCols());
	for (int c1 = 0; c1 < A->getNumRows(); c1++) {
		R->setCell(c1, c1, exp(A->getCell(c1, c1)));
	}
	return R;
}

Matrix* Matrix::zeroMExp(Matrix* A) {
	Matrix* R = new Matrix(A->getNumRows(), A->getNumCols());
	return R;
}

Matrix* Matrix::mExp(Matrix* A, char method, int k) {
	if (A->initialised) {
		// Special Cases
		if (A->isDiagonal()) {
			return diagonalMExp(A);
		}
		else if (A->isZero()) {
			return zeroMExp(A);
		}
		// Ordinary Cases
		else {
			switch (method) {
				default:
					return taylorMExp(A, k);
				case 't':
					return taylorMExp(A, k);
				case 'p':
					return padeMExp(A, k);
			}
		}
	}
	else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

// Booleans

const bool Matrix::isInitialised() {
	return initialised;
}

const bool Matrix::isSquare() {
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

const bool Matrix::isDiagonal() {
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

const bool Matrix::isScalar() {
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

const bool Matrix::isIdentity() {
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

const bool Matrix::isZero() {
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

const double Matrix::getCell(int x, int y) {
	return matrix[x][y];
}

const int Matrix::getNumRows() {
	if (initialised) {
		return numRows;
	}
	else {
		// Error! Cannot determine number of rows in matrix before initialisation
		throw (103);
	}
}

const int Matrix::getNumCols() {
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

void Matrix::setZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				matrix[c1][c2] = 0;
			}
		}
	} else {
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
				} else {
					matrix[c1][c2] = 0;
				}
			}
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

void Matrix::setRandom(double min, double max) {
	if (initialised) {
		std::default_random_engine rng;
		std::uniform_int_distribution<int> gen((int) floor(min), (int) floor(max));
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				int randomI = gen(rng);
				matrix[c1][c2] = randomI;
			}
		}
	}
}

// Output

std::ostream& operator<<(std::ostream& oStream, Matrix* A) {
	if (A->isInitialised()) {
		double cell;
		for (int c1 = 0; c1 < A->getNumRows(); c1++) {
			oStream << "|";
			for (int c2 = 0; c2 < A->getNumCols(); c2++) {
				cell = A->getCell(c1, c2);
				if (abs(cell - (int) (cell) > 0)) {
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
		// Error! Cannot print matrix before initialisation
		throw (102);
	}

	return oStream;
}
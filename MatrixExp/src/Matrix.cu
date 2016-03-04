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

#include "Matrix.cuh"

// Constructors

// Default constructor. Creates an uninitialsed instance of a matrix
Matrix::Matrix() {
	initialised = false;
}
// Creates an instance of a square matrix and initialises it
Matrix::Matrix(int inNumRowsCols) {
	init(inNumRowsCols, inNumRowsCols);
}
// Creates an instance of an (n x m) matrix and initialises it
Matrix::Matrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
}
// Creates an instance of a square matrix and assigns a value to it
Matrix::Matrix(std::vector<std::complex<double>> inMatrix, int inNumRowsCols) {
	init(inNumRowsCols, inNumRowsCols);
	setMatrix(inMatrix);
}
// Creates an instance of an (n x m) matrix and assigns a value to it
Matrix::Matrix(std::vector<std::complex<double>> inMatrix, int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
	setMatrix(inMatrix);
}
// Copy constructor. Copies one instance of a matrix to a new instance
Matrix::Matrix(const Matrix &obj) {
	matrix = obj.matrix;
	numRows = obj.numRows;
	numCols = obj.numCols;
	initialised = obj.initialised;
}
// Initialiser. Resizes the matrix and sets the values to 0
void Matrix::init(int inNumRows, int inNumCols) {
	setNumRows(inNumRows);
	setNumCols(inNumCols);
	matrix.resize(numRows*numCols);
	initialised = true;
	setZero();
}

// Matrix Operations

// Adds two matrices together
Matrix& Matrix::add(Matrix& A, Matrix& B) {
	if (A.initialised && B.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ar == br && ac == bc) {
			Matrix* R = new Matrix(ar, ac);
			for (int c1 = 0; c1 < ar*ac; c1++) {
				R->setCell(c1, A.getCell(c1) + B.getCell(c1));
			}
			return *R;
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
Matrix& Matrix::add(Matrix& A, double x) {
	if (A.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		Matrix *R = new Matrix(ar, ac);
		for (int c1 = 0; c1 < ar*ac; c1++) {
			R->setCell(c1, A.getCell(c1) + x);
		}
		return *R;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Subtracts one matrix from another
Matrix& Matrix::sub(Matrix& A, Matrix& B) {
	if (A.initialised && B.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ar == br && ac == bc) {
			Matrix* R = new Matrix(ar, ac);
			for (int c1 = 0; c1 < ar*ac; c1++) {
				R->setCell(c1, A.getCell(c1) - B.getCell(c1));
			}
			return *R;
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
Matrix& Matrix::sub(Matrix& A, double x) {
	if (A.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		Matrix* R = new Matrix(ar, ac);
		for (int c1 = 0; c1 < ar*ac; c1++) {
			R->setCell(c1, A.getCell(c1) - x);
		}
		return *R;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// Multiplies two matrices together
Matrix& Matrix::mul(Matrix& A, Matrix& B) {
	if (A.initialised && B.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		if (ac == br) {
			Matrix* R = new Matrix(ar, bc);
			std::complex<double> cell;
			for (int c1 = 0; c1 < ar; c1++) {
				for (int c2 = 0; c2 < bc; c2++) {
					cell = 0;
					for (int c3 = 0; c3 < br; c3++) {
						cell += A.getCell(c1, c3) * B.getCell(c3, c2);
					}
					R->setCell(c1, c2, cell);
				}
			}
			return *R;
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
Matrix& Matrix::mul(double x, Matrix& A) {
	if (A.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		Matrix* R = new Matrix(ar, ac);
		for (int c1 = 0; c1 < ar; c1++) {
			for (int c2 = 0; c2 < ac; c2++) {
				R->setCell(c1, c2, A.getCell(c1, c2) * x);
			}
		}
		return *R;
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}
// "Divides" a matrix by another matrix
Matrix& Matrix::div(Matrix& A, Matrix& B) {
	return mul(A, Matrix::inv(B));
}
// "Divides" a matrix by a scalar
Matrix& Matrix::div(Matrix& A, double x) {
	return mul((1 / x), A);
}

Matrix& Matrix::inv(Matrix& A) {
	if (A.initialised) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		if (ar == ac) {
			// Init
			Matrix P(ar, ac*2);
			Matrix* R = new Matrix(ar, ac);
			int c1, c2, c3;
			int n = ac;
			std::complex<double> cell, tmp;
			// Copy A into P (Left side)
			for (c1 = 0; c1 < n; c1++) {
				for (c2 = 0; c2 < n; c2++) {
					P.setCell(c1, c2, A.getCell(c1, c2));
					if (c1 == c2) {
						P.setCell(c1, c2 + n, 1);
					}
				}
			}
			// Pivot P
			for (c1 = n - 1; c1 > 0; c1--) {
				if (P.getCell(c1 - 1, 0).real() < P.getCell(c1, 0).real()) {
					for (c2 = 0; c2 < n * 2; c2++) {
						tmp = P.getCell(c1, c2);
						P.setCell(c1, c2, P.getCell(c1 - 1, c2));
						P.setCell(c1 - 1, c2, tmp);
					}
				}
			}
			// Reduce to diagonal matrix
			for (c1 = 0; c1 < n * 2; c1++) {
				for (c2 = 0; c2 < n; c2++) {
					if (c2 != c1 && c1 < n) {
						tmp = P.getCell(c2, c1) / P.getCell(c1, c1);
						for (c3 = 0; c3 < n * 2; c3++) {
							cell = P.getCell(c2, c3) - (P.getCell(c1, c3) * tmp);
							P.setCell(c2, c3, cell);
						}
					}
				}
			}
			std::cout << P;
			// Reduce to unit matrix
			for (c1 = 0; c1 < n; c1++) {
				tmp = P.getCell(c1, c1);
				for (c2 = 0; c2 < n * 2; c2++) {
					P.setCell(c1, c2, P.getCell(c1, c2) / tmp);
				}
			}
			std::cout << P;
			// Copy P (Right side) to R
			for (c1 = 0; c1 < n; c1++) {
				for (c2 = 0; c2 < n; c2++) {
					R->setCell(c1, c2, P.getCell(c1, c2 + n));
				}
			}
			return *R;
		} else {
			// Error! Cannot find the inverse of this matrix
			throw (205);
		}
	} else {
		// Error! Cannot perform matrix operations before initialisation
		throw (101);
	}
}

Matrix& Matrix::pow(Matrix& A, int x) {
	//if (A.initialised) {
	//	Matrix* R = new Matrix(A.getNumRows(), A.getNumCols());
	//	R->setIdentity();
	//	for (int c1 = 1; c1 <= x; c1++) {
	//		*R = A * *R;
	//	}
	//	return *R;
	//} else {
	//	// Error! Cannot perform matrix operations before initialisation
	//	throw (101);
	//}
	return A;
}

Matrix& Matrix::taylorMExp(Matrix& A, int k) {
	double nfact = 1;
	double coef;
	Matrix An;
	Matrix T;
	Matrix* R = new Matrix(A.getNumRows(), A.getNumCols());
	R->setIdentity();
	for (int n = 1; n <= k; n++) {
		nfact *= n;
		coef = 1.0 / nfact;
		An = Matrix::pow(A, n);
		T = Matrix::mul(coef, An);
		*R = Matrix::add(*R, T);
	}
	return *R;
}

Matrix& Matrix::padeMExp(Matrix& A) {
	// Get params
	params params = getPadeParams(A);
	double s = params.scale;
	int m = params.mVal;
	std::vector<Matrix> p = params.powers;
	std::vector<double> c = getPadeCoefficients(m);
	// Init
	int c1, c2;
	int n = max(A.getNumRows(), A.getNumCols());
	Matrix* R = new Matrix(A);
	Matrix U;
	Matrix V;
	Matrix I(A.getNumRows(), A.getNumCols());
	I.setIdentity();
	// Scaling
	if (s != 0) {
		//A = A / (2.^s);
		//powers = cellfun(@rdivide, powers, ...
		//num2cell(2. ^ (s * (1:length(powers)))), 'UniformOutput', false);
	}
	// Evaluate the Pade approximant.
	if (m == 3 || m == 5 || m == 7 || m == 9) {
		for (c1 = (int) (p.size()) + 2; c1 < m - 1; c1 += 2) { //for (k = strt:2:m-1)
			p[c1] = p[c1-2] * p[2];
		}
		U = c[1] * I;
		V = c[0] * I;
		for (c2 = m; c2 >= 3; c2 -= 2) { //for (j = m : -2 : 3)
			U += (c[c2] * p[c2 - 1]);
			V += (c[c2-1] * p[c2 - 1]);
		}
		U *= A;
	} else if (m == 13) {
		U = A * (p[6] * (c[13] * p[6] + c[11] * p[4] + c[9] * p[2]) + c[7] * p[6] + c[5] * p[4] + c[3] * p[2] + c[1] * I);
		V = p[6] * (c[12] * p[6] + c[10] * p[4] + c[8] * p[2]) + c[6] * p[6] + c[4] * p[4] + c[2] * p[2] + c[0] * I;
	}
	A = (V - U) / (2 * U) + I; //*R = (-U + V) / (U + V);
	//if (recomputeDiags) {
	//	*R = recompute_block_diag(A, *R, blockformat);
	//}
	//// Squaring phase.
	//for (int k = 0; k < s; k++) {
	//	*R = *R**R;
	//	if (recomputeDiags) {
	//		A = 2 * A;
	//		*R = recompute_block_diag(A, *R, blockformat);
	//	}
	//}
	return A;
}

int Matrix::ell(Matrix& A, double coef, int m) {
	//Matrix* scaledA = coef. ^ (1 / (2 * m_val + 1)).*abs(T);
	//alpha = normAm(scaledA, 2 * m_val + 1) / oneNorm(T);
	//t = max(ceil(log2(2 * alpha / eps(class(alpha))) / (2 * m_val)), 0);
	return 0;
}

Matrix::params Matrix::getPadeParams(Matrix& A) {
	params p;
	int d4, d6, d8, d10;
	int eta1, eta3, eta4, eta5;
	std::vector<double> coef, theta;
	p.powers.resize(7);
	p.scale = 0;
	coef = {
		(1 / 100800),
		(1 / 10059033600),
		(1 / 4487938430976000),
		(1 / 9999999999999999999), // Substitute
		(1 / 9999999999999999999)  // Values
		//(1 / 5914384781877411840000LL), // Too long
		//(1 / 113250775606021113483283660800000000)
	};
	theta = {
		3.650024139523051e-008,
		5.317232856892575e-004,
		1.495585217958292e-002,
		8.536352760102745e-002,
		2.539398330063230e-001,
		5.414660951208968e-001,
		9.504178996162932e-001,
		1.473163964234804e+000,
		2.097847961257068e+000,
		2.811644121620263e+000,
		3.602330066265032e+000,
		4.458935413036850e+000,
		5.371920351148152e+000
	};
	p.powers[2] = Matrix::mul(A, A);
	p.powers[4] = Matrix::mul(p.powers[2], p.powers[2]);
	p.powers[6] = Matrix::mul(p.powers[2], p.powers[4]);
	d4 = (int) (std::pow(p.powers[4].getNorm(1), (1 / 4)));
	d6 = (int) (std::pow(p.powers[6].getNorm(1), (1 / 6)));
	eta1 = max(d4, d6);
	if (eta1 <= theta[1] && ell(A, coef[1], 3) == 0) {
		p.mVal = 3;
		return p;
	}
	if (eta1 <= theta[2] && ell(A, coef[2], 5) == 0) {
		p.mVal = 5;
		return p;
	}
	if (A.isSmall()) {
		d8 = (int) ((p.powers[4] * p.powers[4]).getNorm(1));
		d8 = (int) (d8 ^ (1 / 8));
	} else {
		//d8 = normAm(powers[4], 2) ^ (1 / 8);
	}
	eta3 = max(d6, d8);
	if (eta3 <= theta[3] && ell(A, coef[3], 7) == 0) {
		p.mVal = 7;
		return p;
	}
	if (eta3 <= theta[4] && ell(A, coef[4], 9) == 0) {
		p.mVal = 9;
		return p;
	}
	if (A.isSmall()) {
		d10 = (int) (std::pow((p.powers[4] * p.powers[6]).getNorm(1), 1.0 / 10));
	} else {
		//d10 = normAm(powers[2], 5) ^ (1 / 10);
	}
	eta4 = max(d8, d10);
	eta5 = min(eta3, eta4);
	p.scale = max( (int) (ceil(log2(eta5 / theta[5]))), 0);
	//p.scale += ell(pow((A / 2), p.scale), coef[5], 13);
	if (p.scale == INFINITY) {
		//[t, s] = log2(norm(A, 1) / theta.end());
		//s = s - (t == 0.5); //adjust s if normA / theta(end) is a power of 2.
	}
	p.mVal = 13;
	return p;
}

std::vector<double> Matrix::getPadeCoefficients(int m) {
	std::vector<double> coef;
	switch (m) {
		case 3:
			coef = { 120, 60, 12, 1 };
		case 5:
			coef = { 30240, 15120, 3360, 420, 30, 1 };
		case 7:
			coef = { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 };
		case 9:
			coef = { 17643225600, 8821612800, 2075673600, 302702400, 30270240,	2162160, 110880, 3960, 90, 1 };
		case 13:
			coef = { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
	}
	return coef;
}

Matrix& Matrix::diagonalMExp(Matrix& A) {
	Matrix* R = new Matrix(A.getNumRows(), A.getNumCols());
	for (int c1 = 0; c1 < A.getNumRows(); c1++) {
		R->setCell(c1, c1, exp(A.getCell(c1, c1)));
	}
	return *R;
}

Matrix& Matrix::zeroMExp(Matrix& A) {
	Matrix* R = new Matrix(A.getNumRows(), A.getNumCols());
	return *R;
}

Matrix& Matrix::mExp(Matrix& A, char method, int k) {
	if (A.initialised) {
		// Special Cases
		if (A.isDiagonal()) {
			return diagonalMExp(A);
		} else if (A.isZero()) {
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
					return padeMExp(A);
			}
		}
	} else {
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
		} else {
			return false;
		}
	} else {
		// Error! Cannot determine if matrix is square before initialisation
		throw (105);
	}
}

const bool Matrix::isDiagonal() {
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

const bool Matrix::isScalar() {
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

const bool Matrix::isIdentity() {
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

const bool Matrix::isZero() {
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

const bool Matrix::isSmall() {
	return max(numRows, numCols) < 150;
}

// Getters

const std::complex<double> Matrix::getCell(int x) {
	return matrix[x];
}

const std::complex<double> Matrix::getCell(int row, int col) {
	return matrix[row*numCols+col];
}

const int Matrix::getNumRows() {
	if (initialised) {
		return numRows;
	} else {
		// Error! Cannot determine number of rows in matrix before initialisation
		throw (103);
	}
}

const int Matrix::getNumCols() {
	if (initialised) {
		return numCols;
	} else {
		// Error! Cannot determine number of columns in matrix before initialisation
		throw (104);
	}
}

const double Matrix::getNorm(int n) {
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

// Setters

void Matrix::setNumRows(int inNumRows) {
	numRows = inNumRows;
}

void Matrix::setNumCols(int inNumCols) {
	numCols = inNumCols;
}

void Matrix::setMatrix(std::vector<std::complex<double>> inMatrix) {
	matrix = inMatrix;
}

void Matrix::setCell(int x, std::complex<double> value) {
	matrix[x] = value;
}

void Matrix::setCell(int row, int col, std::complex<double> value) {
	matrix[row*numCols+col] = value;
}

void Matrix::setZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows*numCols; c1++) {
			setCell(c1, 0);
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

void Matrix::setRandom(double min, double max) {
	if (initialised) {
		std::default_random_engine rng;
		std::uniform_int_distribution<int> gen((int) floor(min), (int) floor(max));
		for (int c1 = 0; c1 < numRows*numCols; c1++) {
			int randomI = gen(rng);
			setCell(c1, randomI);
		}
	}
}

// General Functions

int max(int x, int y) {
	if (x > y) {
		return x;
	} else {
		return y;
	}
}

double max(double x, double y) {
	if (x > y) {
		return x;
	} else {
		return y;
	}
}

int min(int x, int y) {
	if (x < y) {
		return x;
	} else {
		return y;
	}
}

double min(double x, double y) {
	if (x < y) {
		return x;
	} else {
		return y;
	}
}

// << Operator

std::ostream& operator<<(std::ostream& oStream, Matrix& A) {
	if (A.isInitialised()) {
		std::complex<double> cell;
		for (int c1 = 0; c1 < A.getNumRows(); c1++) {
			oStream << "|";
			for (int c2 = 0; c2 < A.getNumCols(); c2++) {
				cell = A.getCell(c1, c2);
				if (abs(cell.real() - (int) (cell.real()) != 0)) {
					// Decimal
					if (cell.imag() != 0) {
						oStream << " " << std::setprecision(3) << std::fixed << cell;
					} else {
						oStream << " " << std::setprecision(3) << std::fixed << cell.real();
					}
				} else {
					// Integer
					if (cell.imag() != 0) {
						oStream << " " << std::setprecision(0) << std::fixed << cell;
					} else {
						oStream << " " << std::setprecision(0) << std::fixed << cell.real();
					}
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

// + Operator

Matrix& operator+(Matrix& A, Matrix& B) {
	return Matrix::add(A, B);
}

Matrix& operator+(Matrix& A, double B) {
	return Matrix::add(A, B);
}

Matrix& operator+(double A, Matrix& B) {
	return Matrix::add(B, A);
}

// += Operator

Matrix& operator+=(Matrix& A, Matrix& B) {
	return Matrix::add(A, B);
}

Matrix& operator+=(Matrix& A, double B) {
	return Matrix::add(A, B);
}

// - Operator

Matrix& operator-(Matrix& A, Matrix& B) {
	return Matrix::sub(A, B);
}

Matrix& operator-(Matrix& A, double B) {
	return Matrix::sub(A, B);
}

// -= Operator

Matrix& operator-=(Matrix& A, Matrix& B) {
	return Matrix::sub(A, B);
}

Matrix& operator-=(Matrix& A, double B) {
	return Matrix::sub(A, B);
}

// * Operator

Matrix& operator*(Matrix& A, Matrix& B) {
	return Matrix::mul(A, B);
}

Matrix& operator*(Matrix& A, double B) {
	return Matrix::mul(B, A);
}

Matrix& operator*(double A, Matrix& B) {
	return Matrix::mul(A, B);
}

// *= Operator

Matrix& operator*=(Matrix& A, Matrix& B) {
	return Matrix::mul(A, B);
}

Matrix& operator*=(Matrix& A, double B) {
	return Matrix::mul(B, A);
}

// / Operator

Matrix& operator/(Matrix& A, Matrix& B) {
	return Matrix::div(A, B);
}

Matrix& operator/(Matrix& A, double B) {
	return Matrix::div(A, B);
}

// /= Operator

Matrix& operator/=(Matrix& A, Matrix& B) {
	return Matrix::div(A, B);
}

Matrix& operator/=(Matrix& A, double B) {
	return Matrix::div(A, B);
}

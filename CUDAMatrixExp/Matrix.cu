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
#include "Matrix.cuh"

// KERNELS

__global__ void cudaAdd(double* A, double* B, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] + B[row * n + col];
	}
	__syncthreads();
}

__global__ void cudaSub(double* A, double* B, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] - B[row * n + col];
	}
	__syncthreads();
}

__global__ void cudaMul(double* A, double* B, double* R, int n) {
	double sum = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		for (int i = 0; i < n; i++) {
			sum += A[row * n + i] * B[i * n + col];
		}
	}
	R[row * n + col] = sum;
	__syncthreads();
}

__global__ void cudaCholeskyInv(double* R, int k, int n, int stride) {
	unsigned int j;
	int i = blockIdx.x + (k + 1);
	int offset = i;
	int jstart = threadIdx.x + offset;
	int jstep = stride;
	int jtop = n - 1;
	int jbottom = i;
	for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
		R[i * n + j] -= R[k * n + i] * R[k * n + j];
	}
}

__global__ void cudaCholeskyDiv(double* R, int k, int n, int stride) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j;
	if (tx == 0) {
		R[k * n + k] = sqrt(R[k * n + k]);
	}
	int offset = (k + 1);
	int jstart = threadIdx.x + offset;
	int jstep = stride;
	int jtop = n - 1;
	int jbottom = (k + 1);
	if (blockIdx.x == 0) {
		for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
			R[k * n + j] /= R[k * n + k];
		}
	}
}

__global__ void cudaGaussJordanInv(double* A, double* R, int n, int i) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double P;

	if (col < n && row < n) {
		if (col > i) {
			if (row > i) {
				P = A[col * n + i] / A[i*n + i];
				R[col * n + row] -= R[i*n + row] * P;
				A[col*n + row] -= A[i*n + row] * P;
			}
			__syncthreads();
		}
	}
} 

__global__ void cudaGuassJordanDev(double* A, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < n && row < n) {
		if (A[col * n + col] != 0) {
			R[col * n + row] /= A[col * n + col];
			A[col * n + row] /= A[col * n + col];
		}
	} 
	__syncthreads();
}

// INTERNAL METHODS

// Allocate memory on the host and device
void CUDAMatrix::alloc() {
	h_matrix = (double*) malloc(size);
	cudaError_t result = cudaMalloc((void**) &d_matrix, size);
	if (result != cudaSuccess) {
		throw std::runtime_error("Failed to allocate device memory");
	}
}
// Deallocate memory on the host and device
void CUDAMatrix::dealloc() {
	free(h_matrix);
	cudaError_t result = cudaFree(d_matrix);
	if (result != cudaSuccess) {
		throw std::runtime_error("Failed to free device memory");
	}
}
// Sync host matrix with device matrix
void CUDAMatrix::syncHost() {
	if (isInitialised()) {
		cudaError_t result = cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to allocate device memory");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Sync device matrix with host matrix
void CUDAMatrix::syncDevice() {
	if (isInitialised()) {
		cudaError_t result = cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to allocate device memory");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Start CUDA timer
CUDAMatrix::cudaTimer CUDAMatrix::startTimer() {
	cudaTimer t;
	cudaEventCreate(&t.start);
	cudaEventCreate(&t.stop);
	cudaEventRecord(t.start, 0);
	return t;
}
// Stop CUDA timer
float CUDAMatrix::stopTimer(CUDAMatrix::cudaTimer t) {
	float time;
	cudaEventRecord(t.stop, 0);
	cudaEventSynchronize(t.stop);
	cudaEventElapsedTime(&time, t.start, t.stop);
	cudaEventDestroy(t.start);
	cudaEventDestroy(t.stop);
	return time;
}
// Get CUDA params
CUDAMatrix::cudaParams CUDAMatrix::getCUDAParams(int rows, int cols) {
	cudaParams cp;
	cp.tpb = dim3(rows, cols);
	cp.bpg = dim3(1, 1);
	if (rows*cols > 512) {
		cp.tpb.x = 512;
		cp.tpb.y = 512;
		cp.bpg.x = (int) (ceil(double(rows) / double(cp.tpb.x)));
		cp.bpg.y = (int) (ceil(double(cols) / double(cp.tpb.y)));
	}
	return cp;
}
// Get the pade parameters
CUDAMatrix::padeParams CUDAMatrix::getPadeParams(CUDAMatrix& A) {
	padeParams p;
	//	int d4, d6, d8, d10;
	//	int eta1, eta3, eta4, eta5;
	//	std::vector<double> coef, theta;
	//	p.powers.resize(7);
	//	p.scale = 0;
	//	coef = {
	//		(1 / 100800),
	//		(1 / 10059033600),
	//		(1 / 4487938430976000),
	//		(1 / 9999999999999999999), // Substitute
	//		(1 / 9999999999999999999)  // Values
	//		//(1 / 5914384781877411840000LL), // Too long
	//		//(1 / 113250775606021113483283660800000000)
	//	};
	//	theta = {
	//		3.650024139523051e-008,
	//		5.317232856892575e-004,
	//		1.495585217958292e-002,
	//		8.536352760102745e-002,
	//		2.539398330063230e-001,
	//		5.414660951208968e-001,
	//		9.504178996162932e-001,
	//		1.473163964234804e+000,
	//		2.097847961257068e+000,
	//		2.811644121620263e+000,
	//		3.602330066265032e+000,
	//		4.458935413036850e+000,
	//		5.371920351148152e+000
	//	};
	//	p.powers[2] = CUDAMatrix::mul(A, A);
	//	p.powers[4] = CUDAMatrix::mul(p.powers[2], p.powers[2]);
	//	p.powers[6] = CUDAMatrix::mul(p.powers[2], p.powers[4]);
	//	d4 = (int) (std::pow(p.powers[4].getNorm(1), (1 / 4)));
	//	d6 = (int) (std::pow(p.powers[6].getNorm(1), (1 / 6)));
	//	eta1 = utils::max(d4, d6);
	//	if (eta1 <= theta[1] && ell(A, coef[1], 3) == 0) {
	//		p.mVal = 3;
	//		return p;
	//	}
	//	if (eta1 <= theta[2] && ell(A, coef[2], 5) == 0) {
	//		p.mVal = 5;
	//		return p;
	//	}
	//	if (A.isSmall()) {
	//		d8 = (int) ((p.powers[4] * p.powers[4]).getNorm(1));
	//		d8 = (int) (d8 ^ (1 / 8));
	//	} else {
	//		//d8 = normAm(powers[4], 2) ^ (1 / 8);
	//	}
	//	eta3 = utils::max(d6, d8);
	//	if (eta3 <= theta[3] && ell(A, coef[3], 7) == 0) {
	//		p.mVal = 7;
	//		return p;
	//	}
	//	if (eta3 <= theta[4] && ell(A, coef[4], 9) == 0) {
	//		p.mVal = 9;
	//		return p;
	//	}
	//	if (A.isSmall()) {
	//		d10 = (int) (std::pow((p.powers[4] * p.powers[6]).getNorm(1), 1.0 / 10));
	//	} else {
	//		//d10 = normAm(powers[2], 5) ^ (1 / 10);
	//	}
	//	eta4 = utils::max(d8, d10);
	//	eta5 = utils::min(eta3, eta4);
	//	p.scale = utils::max((int) (ceil(log2(eta5 / theta[5]))), 0);
	//	//p.scale += ell(pow((A / 2), p.scale), coef[5], 13);
	//	if (p.scale == INFINITY) {
	//		//[t, s] = log2(norm(A, 1) / theta.end());
	//		//s = s - (t == 0.5); //adjust s if normA / theta(end) is a power of 2.
	//	}
	//	p.mVal = 13;
	return p;
}
// Get the pade coefficients
int* CUDAMatrix::getPadeCoefficients(int m) {
//	switch (m) {
//		case 3:
//			return new int[4] { 120, 60, 12, 1 };
//		case 5:
//			return new int[6] { 30240, 15120, 3360, 420, 30, 1 };
//		case 7:
//			return new int[8] { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 };
//		case 9:
//			return new int[10] { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 };
//		case 13:
//			return new int[14] { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
//		default:
//			throw std::runtime_error("Invalid m value");
//	}
	throw;
}

// CONSTRUCTORS

// Default constructor. Creates an uninitialsed instance of a matrix
CUDAMatrix::CUDAMatrix() {
	initialised = false;
}
// Creates an instance of a square matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRowsCols) {
	init(inNumRowsCols, inNumRowsCols);
	setMatrix('i');
}
// Creates an instance of an (n x m) matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
	setMatrix('i');
}
// Creates an instance of a square matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRowsCols, std::initializer_list<double> inMatrix) {
	if (inMatrix.size() == inNumRowsCols*inNumRowsCols) {
		init(inNumRowsCols, inNumRowsCols);
		setMatrix(inMatrix);
	} else {
		throw std::runtime_error("Initialiser-list size does not match matrix size");
	}
}
// Creates an instance of an (n x m) matrix and assigns a value to it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols, std::initializer_list<double> inMatrix) {
	if (inMatrix.size() == inNumRows*inNumCols) {
		init(inNumRows, inNumCols);
		setMatrix(inMatrix);
	} else {
		throw std::runtime_error("Initialiser-list size does not match matrix size");
	}
}
// Copy constructor
CUDAMatrix::CUDAMatrix(const CUDAMatrix &obj) {
	if (obj.initialised) {
		h_matrix = obj.h_matrix;
		d_matrix = obj.d_matrix;
		numRows = obj.numRows;
		numCols = obj.numCols;
		numEls = obj.numEls;
		size = obj.size;
		initialised = obj.initialised;
	} else {
		throw std::runtime_error("Cannot copy uninitialised matrix");
	}
}
// Initialiser
void CUDAMatrix::init(int inNumRows, int inNumCols) {
	numRows = inNumRows;
	numCols = inNumCols;
	numEls = inNumRows*inNumCols;
	size = sizeof(double) *numEls;
	alloc();
	initialised = true;
}
// Destructor
CUDAMatrix::~CUDAMatrix() {
	dealloc();
}

// KERNEL CALLS

float CUDAMatrix::tra(CUDAMatrix& A, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ac == rr) {
			int c1, c2;
			cudaTimer t = startTimer();
			for (c1 = 0; c1 < A.getNumRows(); c1++) {
				for (c2 = 0; c2 < A.getNumCols(); c2++) {
					R.setCell(c1, c2, A.getCell(c2, c1));
				}
			}
			float time = stopTimer(t);
			R.syncDevice();
			return time;
		} else {
			throw std::runtime_error("Transpose matrix is the wrong size");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

float CUDAMatrix::add(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			cudaParams cp = getCUDAParams(ar, ac);
			cudaTimer t = startTimer();

			cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());

			float time = stopTimer(t);
			R.syncHost();
			return time;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

float CUDAMatrix::sub(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			cudaParams cp = getCUDAParams(ar, ac);
			cudaTimer t = startTimer();

			cudaSub KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());

			float time = stopTimer(t);
			R.syncHost();
			return time;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

float CUDAMatrix::mul(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			cudaParams cp = getCUDAParams(ar, ac);
			cudaTimer t = startTimer();

			cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());

			float time = stopTimer(t);
			R.syncHost();
			return time;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

float CUDAMatrix::inv(CUDAMatrix& A, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			int c1, numBlocks;
			int n = ar;
			int threadsPerBlock = 256;
			int stride = 256;
			for (c1 = 0; c1 < n*n; c1++) {
				R.setCell(c1, A.getCell(c1));
			}
			R.syncDevice();
			cudaTimer t = startTimer();
			for (c1 = 0; c1 < n; c1++) {
				numBlocks = (n - 1) - (c1 + 1) + 1;
				if (numBlocks <= 0) {
					numBlocks = 1;
				}
				dim3 threadBlock(threadsPerBlock, 1, 1);
				dim3 grid(numBlocks, 1);
				cudaCholeskyDiv KERNEL_ARGS2(grid, threadBlock) (R.d_matrix, c1, n, stride);
				cudaCholeskyInv KERNEL_ARGS2(grid, threadBlock) (R.d_matrix, c1, n, stride);

				cudaThreadSynchronize();
			}
			cudaThreadSynchronize();

			R.syncHost();
			int i, j;
			for (i = 0; i < n; i++) {
				for (j = 0; j < i; j++) {
					R.h_matrix[i * n + j] = 0.0;
				}
			}
			R.syncDevice();

			//for (int i = 0; i < n; i++) {
			//	cudaGaussJordanInv KERNEL_ARGS2(blocksPerGrid, threadsPerBlock) (A.d_matrix, R.d_matrix, n, i);
			//}
			//cudaGaussJordanDev KERNEL_ARGS2(blocksPerGrid, threadsPerBlock) (A.d_matrix, R.d_matrix, n);
			float time = stopTimer(t);
			return time;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// BOOLEANS

bool CUDAMatrix::isInitialised() {
	return initialised;
}

// SETTERS

void CUDAMatrix::setCell(int row, int col, double val) {
	if (isInitialised()) {
		h_matrix[numCols*row + col] = val;
		syncDevice();
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setCell(int i, double val) {
	if (isInitialised()) {
		h_matrix[i] = val;
		syncDevice();
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(int val) {
	if (isInitialised()) {
		for (int c1 = 0; c1 < getNumEls(); c1++) {
			h_matrix[c1] = val;
		}
		syncDevice();
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(const char val) {
	if (isInitialised()) {
		if (val == 'i') {
			int row, col;
			for (int c1 = 0; c1 < getNumEls(); c1++) {
				row = getCurRow(c1);
				col = getCurCol(c1);
				if (row == col) {
					h_matrix[c1] = 1;
				} else {
					h_matrix[c1] = 0;
				}
			}
		} else {
			throw std::runtime_error("Parameter is undefined");
		}
		syncDevice();
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(double* inMatrix) {
	if (isInitialised()) {
		memcpy(&h_matrix, inMatrix, size);
		syncDevice();
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(std::initializer_list<double> inMatrix) {
	if (isInitialised()) {
		if (inMatrix.size() == getNumEls()) {
			std::copy(inMatrix.begin(), inMatrix.end(), h_matrix);
			syncDevice();
		} else {
			throw std::runtime_error("Initialiser-list size does not match matrix size");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// GETTERS

int CUDAMatrix::getCurRow(int i) {
	if (isInitialised()) {
		return (int) (floor(i / numCols));
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

int CUDAMatrix::getCurCol(int i) {
	if (isInitialised()) {
		return (int) (i - (numCols*getCurRow(i)));
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

double CUDAMatrix::getCell(int row, int col) {
	if (isInitialised()) {
		return h_matrix[row*numCols + col];
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

double CUDAMatrix::getCell(int i) {
	if (isInitialised()) {
		return h_matrix[i];
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

double* CUDAMatrix::getMatrix() {
	if (isInitialised()) {
		return h_matrix;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

int CUDAMatrix::getNumRows() {
	if (isInitialised()) {
		return numRows;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

int CUDAMatrix::getNumCols() {
	if (isInitialised()) {
		return numCols;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

int CUDAMatrix::getNumEls() {
	if (isInitialised()) {
		return numEls;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

size_t CUDAMatrix::getSize() {
	if (isInitialised()) {
		return size;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// OPERATOR OVERRIDES

// <<
std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A) {
	if (A.isInitialised()) {
		double cell;
		int c1, c2, length, maxLength = 0, precision = 0;
		for (c1 = 0; c1 < A.getNumEls(); c1++) {
			// Get precision
			cell = A.getCell(c1);
			if (cell - (int) cell > 0.0) {
				precision = 3;
			}
			// Get maximum number length
			length = Utils::getNumDigits(cell);
			if (length > maxLength) {
				maxLength = length;
			}
		}
		for (c1 = 0; c1 < A.getNumEls(); c1++) {
			cell = A.getCell(c1);
			// Remove negative zeros
			if (cell == 0) {
				cell = 0;
			}
			oStream << "| ";
			// Add whitespace if shorter than maxLength
			length = Utils::getNumDigits(cell);
			for (c2 = 0; c2 < (maxLength - length); c2++) {
				oStream << " ";
			}
			// Output number
			oStream << std::setprecision(precision) << std::fixed << cell << " ";
			// Output new line if row end reached
			if (A.getCurRow(c1 + 1) > A.getCurRow(c1)) {
				oStream << "|" << std::endl;
			}
		}
		return oStream;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// UTILS

int Utils::getNumDigits(double x) {
	int length;
	if (x > 1 || x < -1) {
		length = (int) (floor(log10(abs(x))) + 1);
	} else {
		length = 1;
	}
	if (x < 0) {
		length++;
	}
	return length;
}
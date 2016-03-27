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

__global__ void cudaAdd(double* A, double* B, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] + B[row * n + col];
	}
	__syncthreads();
}

__global__ void cudaAddScalar(double* A, double scalar, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] + scalar;
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

__global__ void cudaSubScalar(double* A, double scalar, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] - scalar;
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

__global__ void cudaMulScalar(double* A, double scalar, double* R, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		R[row * n + col] = A[row * n + col] * scalar;
	}
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

// MEMORY HANDLERS

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

// CUDA STUFF

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

// INTERNAL PADE APPROXIMATION CODE

// Get the pade parameters
CUDAMatrix::padeParams CUDAMatrix::getPadeParams(CUDAMatrix& A) {
	// Init
	int d4, d6, d8, d10;
	int eta1, eta3, eta4, eta5;
	int ar = A.getNumRows();
	int ac = A.getNumCols();
	std::vector<double> theta;
	std::vector<double> coef;
	// Init P;
	padeParams p;
	p.pow.resize(11);
	p.scale = 0;
	// Get coefficients and theta values
	coef = {
		(1 / 100800),
		(1 / 10059033600),
		(1 / 4487938430976000),
		(1 / 9999999999999999999), // Substitute
		(1 / 9999999999999999999)  // Values
		//(1 / 5914384781877411840000),
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
	// Get powers of A
	p.pow[2] = new CUDAMatrix(ar, ac);
	p.pow[4] = new CUDAMatrix(ar, ac);
	p.pow[6] = new CUDAMatrix(ar, ac);
	p.pow[8] = new CUDAMatrix(ar, ac);
	p.pow[10] = new CUDAMatrix(ar, ac);
	cudaParams cp = getCUDAParams(A.getNumRows(), A.getNumCols());
	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, A.d_matrix, p.pow[2]->d_matrix, ar);
	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[2]->d_matrix, p.pow[4]->d_matrix, ar);
	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[4]->d_matrix, p.pow[6]->d_matrix, ar);
	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[4]->d_matrix, p.pow[8]->d_matrix, ar);
	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[6]->d_matrix, p.pow[10]->d_matrix, ar);

	// NOT IDEAL .. PERFORM GETNORM ON DEVICE IF POSSIBLE. THIS MEANS SYNCING BETWEEN HOST AND DEVICE IS UNNECESSARY
	p.pow[2]->syncHost();
	p.pow[4]->syncHost();
	p.pow[6]->syncHost();
	p.pow[8]->syncHost();
	p.pow[10]->syncHost();
	////

	// Get norms
	d4 = (int) (std::pow(p.pow[4]->getNorm(1), (1 / 4)));
	d6 = (int) (std::pow(p.pow[6]->getNorm(1), (1 / 6)));
	eta1 = utils::max(d4, d6);
	if (eta1 <= theta[1] && ell(A, coef[1], 3) == 0) {
		p.mVal = 3;
		return p;
	}
	if (eta1 <= theta[2] && ell(A, coef[2], 5) == 0) {
		p.mVal = 5;
		return p;
	}
	if (true) {//(A.isSmall()) {
		d8 = (int) (p.pow[8]->getNorm(1));
		d8 = (int) (d8 ^ (1 / 8));
	} else {
		//d8 = normAm(powers[4], 2) ^ (1 / 8);
	}
	eta3 = utils::max(d6, d8);
	if (eta3 <= theta[3] && ell(A, coef[3], 7) == 0) {
		p.mVal = 7;
		return p;
	}
	if (eta3 <= theta[4] && ell(A, coef[4], 9) == 0) {
		p.mVal = 9;
		return p;
	}
	if (true) { //(A.isSmall()) {
		d10 = (int) (std::pow(p.pow[10]->getNorm(1), 1.0 / 10));
	} else {
		//d10 = (int) (std::pow(normAm(powers[2], 5), 1.0 / 10));
	}
	eta4 = utils::max(d8, d10);
	eta5 = utils::min(eta3, eta4);
	//p.scale = utils::max((int) (ceil(log2(eta5 / theta[5]))), 0);
	//p.scale += ell(pow((A / 2), p.scale), coef[5], 13);
	//if (p.scale == INFINITY) {
		//[t, s] = log2(norm(A, 1) / theta.end());
		//s = s - (t == 0.5); //adjust s if normA / theta(end) is a power of 2.
	//}
	p.mVal = 13;
	return p;
}
// Something to do with pade scaling
int CUDAMatrix::ell(CUDAMatrix& A, double coef, int m) {
	//Matrix* scaledA = coef. ^ (1 / (2 * m_val + 1)).*abs(T);
	//alpha = normAm(scaledA, 2 * m_val + 1) / oneNorm(T);
	//t = max(ceil(log2(2 * alpha / eps(class(alpha))) / (2 * m_val)), 0);
	return 0;
}
// Get the pade coefficients
std::vector<double> CUDAMatrix::getPadeCoefficients(int m) {
	switch (m) {
		case 3:
			return { 120, 60, 12, 1 };
		case 5:
			return { 30240, 15120, 3360, 420, 30, 1 };
		case 7:
			return { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 };
		case 9:
			return { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 };
		case 13:
			return { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
		default:
			throw std::runtime_error("Invalid m value");
	}
}

// CONSTRUCTORS

// Default constructor. Creates an uninitialsed instance of a matrix
CUDAMatrix::CUDAMatrix() {
	initialised = false;
}
// Creates an instance of a square matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRowsCols) {
	init(inNumRowsCols, inNumRowsCols);
	setMatrix(0);
}
// Creates an instance of an (n x m) matrix and initialises it
CUDAMatrix::CUDAMatrix(int inNumRows, int inNumCols) {
	init(inNumRows, inNumCols);
	setMatrix(0);
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

// MATRIX OPERATIONS

CUDATimer CUDAMatrix::add(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			A.syncDevice();
			B.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;

			t.start();
			cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::add(CUDAMatrix& A, double scalar, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			A.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;
			
			t.start();
			cudaAddScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, scalar, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::sub(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			A.syncDevice();
			B.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;
			
			t.start();
			cudaSub KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::sub(CUDAMatrix& A, double scalar, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			A.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;

			t.start();
			cudaSubScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, scalar, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::mul(CUDAMatrix& A, CUDAMatrix& B, CUDAMatrix& R) {
	if (A.isInitialised() && B.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int br = B.getNumRows();
		int bc = B.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == br && br == bc && bc == rr && rr == rc) {
			A.syncDevice();
			B.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;
			
			t.start();
			cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, B.d_matrix, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::mul(CUDAMatrix& A, double scalar, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			A.syncDevice();

			cudaParams cp = getCUDAParams(ar, ac);
			CUDATimer t;
			
			t.start();
			cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, scalar, R.d_matrix, A.getNumRows());
			t.stop();

			R.syncHost();
			return t;
		} else {
			throw std::runtime_error("Matrix sizes do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

//CUDATimer CUDAMatrix::inv(CUDAMatrix& A, CUDAMatrix& R) {
//	if (A.isInitialised() && R.isInitialised()) {
//		int ar = A.getNumRows();
//		int ac = A.getNumCols();
//		int rr = R.getNumRows();
//		int rc = R.getNumCols();
//		if (ar == ac && ac == rr && rr == rc) {
//			A.syncDevice();
//
//			CUDATimer t;
//			int c1, numBlocks;
//			int n = ar;
//			int threadsPerBlock = 256;
//			int stride = 256;
//			for (c1 = 0; c1 < n*n; c1++) {
//				R.setCell(c1, A.getCell(c1));
//			}
//			R.syncDevice();
//			
//			t.start();
//			for (c1 = 0; c1 < n; c1++) {
//				numBlocks = (n - 1) - (c1 + 1) + 1;
//				if (numBlocks <= 0) {
//					numBlocks = 1;
//				}
//				dim3 threadBlock(threadsPerBlock, 1, 1);
//				dim3 grid(numBlocks, 1);
//				cudaCholeskyDiv KERNEL_ARGS2(grid, threadBlock) (R.d_matrix, c1, n, stride);
//				cudaCholeskyInv KERNEL_ARGS2(grid, threadBlock) (R.d_matrix, c1, n, stride);
//
//				cudaThreadSynchronize();
//			}
//			cudaThreadSynchronize();
//
//			R.syncHost();
//			int i, j;
//			for (i = 0; i < n; i++) {
//				for (j = 0; j < i; j++) {
//					R.h_matrix[i * n + j] = 0.0;
//				}
//			}
//			t.stop();
//
//			R.syncDevice();
//
//			//for (int i = 0; i < n; i++) {
//			//	cudaGaussJordanInv KERNEL_ARGS2(blocksPerGrid, threadsPerBlock) (A.d_matrix, R.d_matrix, n, i);
//			//}
//			//cudaGaussJordanDev KERNEL_ARGS2(blocksPerGrid, threadsPerBlock) (A.d_matrix, R.d_matrix, n);
//			return t;
//		} else {
//			throw std::runtime_error("Matrix sizes do not match");
//		}
//	} else {
//		throw std::runtime_error("Cannot perform matrix operations before initialisation");
//	}
//}

CUDATimer CUDAMatrix::inv(CUDAMatrix& A, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			CUDATimer t;
			t.start();
			// Init
			CUDAMatrix P(ar, ac * 2);
			int c1, c2, c3;
			int n = ac;
			double cell, tmp;
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
				if (P.getCell(c1 - 1, 0) < P.getCell(c1, 0)) {
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
			// Reduce to unit matrix
			for (c1 = 0; c1 < n; c1++) {
				tmp = P.getCell(c1, c1);
				for (c2 = 0; c2 < n * 2; c2++) {
					P.setCell(c1, c2, P.getCell(c1, c2) / tmp);
				}
			}
			// Copy P (Right side) to R
			for (c1 = 0; c1 < n; c1++) {
				for (c2 = 0; c2 < n; c2++) {
					R.setCell(c1, c2, P.getCell(c1, c2 + n));
				}
			}
			t.stop();
			R.syncDevice();
			return t;
		} else {
			throw std::runtime_error("Cannot find the inverse of this matrix");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::tra(CUDAMatrix& A, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ac == rr) {
			A.syncDevice();

			int c1, c2;
			CUDATimer t;
			
			t.start();
			for (c1 = 0; c1 < A.getNumRows(); c1++) {
				for (c2 = 0; c2 < A.getNumCols(); c2++) {
					R.setCell(c1, c2, A.getCell(c2, c1));
				}
			}
			t.stop();

			R.syncDevice();
			return t;
		} else {
			throw std::runtime_error("Transpose matrix is the wrong size");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

CUDATimer CUDAMatrix::exp(CUDAMatrix& A, CUDAMatrix& R) {
	if (A.isInitialised() && R.isInitialised()) {
		int ar = A.getNumRows();
		int ac = A.getNumCols();
		int rr = R.getNumRows();
		int rc = R.getNumCols();
		if (ar == ac && ac == rr && rr == rc) {
			A.syncDevice();
			CUDATimer t;
			int c1, c2;
			int n = utils::max(ar, ac);
			// Special Cases
			if (A.isDiagonal() && false) {											// REMOVE && FALSE WHEN FINISHED TESTING
				t.start();
				for (c1 = 0; c1 < n; c1++) {
					R.setCell(c1, c1, std::exp(A.getCell(c1, c1)));
				}
				t.stop();
				R.syncDevice();
			} else if (A.isZero()) {
				t.start();
				R.setMatrix(0);
				t.stop();
				R.syncDevice();
			// Normal Case
			} else {
				// Create Matrices
				CUDAMatrix U(ar, ac);
				CUDAMatrix V(ar, ac);
				CUDAMatrix I(ar, ac); // Identity
				CUDAMatrix T(ar, ac); // Tally
				CUDAMatrix TMP(ar, ac); // Temporary
				I.setIdentity();
				I.syncDevice();
				// Get CUDA params
				cudaParams cp = getCUDAParams(ar, ac);
				// Get Pade params
				padeParams p = getPadeParams(A);
				double s = p.scale;
				int m = p.mVal;
				std::vector<CUDAMatrix*> pow = p.pow;
				// Get Pade coefficients
				std::vector<double> c = getPadeCoefficients(m);
				// OUTPUT
				std::cout << "s = " << s << std::endl;
				std::cout << "m = " << m << std::endl;
				// Start timer
				t.start();
				// Scaling
				//if (s != 0) {
				//	A = A / (2.^s);
				//	powers = cellfun(@rdivide, powers, ...
				//	num2cell(2. ^ (s * (1:length(powers)))), 'UniformOutput', false);
				//}
				// Approximation

				if (m == 3 || m == 5 || m == 7 || m == 9) {
					for (c1 = (int) (pow.size()) + 2; c1 < m - 1; c1 += 2) { //for (k = strt:2:m-1)
						cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c1 - 2]->d_matrix, pow[2]->d_matrix, pow[c1]->d_matrix, n);
					}
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[1], U.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[0], V.d_matrix, n);
					for (c2 = m; c2 >= 3; c2 -= 2) { //for (j = m : -2 : 3)
						cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c2 - 1]->d_matrix, c[c2], TMP.d_matrix, n);
						cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, TMP.d_matrix, U.d_matrix, n);
						cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c2 - 1]->d_matrix, c[c2-1], TMP.d_matrix, n);
						cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (V.d_matrix, TMP.d_matrix, V.d_matrix, n);
					}
					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, A.d_matrix, U.d_matrix, n);
				} else if (m == 13) {
					// This is the equivellent of .. 
					// U = A * (p[6] * (c[13] * p[6] + c[11] * p[4] + c[9] * p[2]) + c[7] * p[6] + c[5] * p[4] + c[3] * p[2] + c[1] * I);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[13], T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[11], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[9], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, pow[6]->d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[7], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[5], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[3], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[1], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, T.d_matrix, U.d_matrix, n);
					// This is the equivellent of ..
					//V = p[6] * (c[12] * p[6] + c[10] * p[4] + c[8] * p[2]) + c[6] * p[6] + c[4] * p[4] + c[2] * p[2] + c[0] * I;
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[12], T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[10], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[8], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, pow[6]->d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[6], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[4], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[2], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[0], TMP.d_matrix, n);
					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, V.d_matrix, n);
					// TESTING
					A.syncHost();
					U.syncHost();
					V.syncHost();
					I.syncHost();
					T.syncHost();
					TMP.syncHost();
					std::cout << "A" << A << "U" << U << "V" << V << "I" << I << "T" << T << "TMP" << TMP;
					/////////
				}
				// This is the equivellent of ..
				// R = (V - U) / (2 * U) + I;  ||?? R = (-U + V) / (U + V);
				cudaSub KERNEL_ARGS2(cp.bpg, cp.tpb) (V.d_matrix, U.d_matrix, T.d_matrix, n);
				cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, 2, TMP.d_matrix, n);
				//cudaInv KERNEL_ARGS2(cp.bpg, cp.tpb) (TMP.d_matrix, TMP.d_matrix, n); // TEMP CODE BELOW
				TMP.syncHost();
				CUDAMatrix::inv(TMP, TMP);
				TMP.syncDevice();
				//
				cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
				cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, I.d_matrix, R.d_matrix, n);
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
				t.stop();
				R.syncHost();
			}
			return t;
		} else {
			throw std::runtime_error("Matrix sizez do not match");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// BOOLEANS

bool CUDAMatrix::isInitialised() {
	return initialised;
}
// Check if a matrix is square
bool CUDAMatrix::isSquare() {
	if (initialised) {
		if (numCols == numRows) {
			return true;
		} else {
			return false;
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Check if a matrix is diagonal
bool CUDAMatrix::isDiagonal() {
	if (initialised) {
		if (!isSquare()) {
			return false;
		}
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (c1 != c2 && getCell(c1, c2) != 0) {
					return false;
				}
			}
		}
		return true;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Check if a matrix is an identity matrix
bool CUDAMatrix::isIdentity() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if ((c1 != c2 && getCell(c1, c2) != 0) || (c1 == c2 && getCell(c1, c2) != 1)) {
					return false;
				}
			}
		}
		return true;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Check if a matrix is a zero matrix
bool CUDAMatrix::isZero() {
	if (initialised) {
		for (int c1 = 0; c1 < numRows; c1++) {
			for (int c2 = 0; c2 < numCols; c2++) {
				if (getCell(c1, c2) != 0) {
					return false;
				}
			}
		}
		return true;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
// Check if a matrix is "small"
bool CUDAMatrix::isSmall() {
	return utils::max(numRows, numCols) < 150;
}

// SETTERS

void CUDAMatrix::setCell(int row, int col, double val) {
	if (isInitialised()) {
		h_matrix[numCols*row + col] = val;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setCell(int i, double val) {
	if (isInitialised()) {
		h_matrix[i] = val;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(int val) {
	if (isInitialised()) {
		for (int c1 = 0; c1 < getNumEls(); c1++) {
			h_matrix[c1] = val;
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(double* inMatrix) {
	if (isInitialised()) {
		memcpy(&h_matrix, inMatrix, size);
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setMatrix(std::initializer_list<double> inMatrix) {
	if (isInitialised()) {
		if (inMatrix.size() == getNumEls()) {
			std::copy(inMatrix.begin(), inMatrix.end(), h_matrix);
		} else {
			throw std::runtime_error("Initialiser-list size does not match matrix size");
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setIdentity() {
	if (isInitialised()) {
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
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setRandomDouble(double min, double max) {
	if (isInitialised()) {
		double r;
		std::default_random_engine rng(time(0));
		std::uniform_real_distribution<double> gen(min, max);
		for (int c1 = 0; c1 < numEls; c1++) {
			r = gen(rng);
			setCell(c1, r);
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

void CUDAMatrix::setRandomInt(int min, int max) {
	if (isInitialised()) {
		int r;
		std::default_random_engine rng(time(0));
		std::uniform_int_distribution<int> gen(min, max);
		for (int c1 = 0; c1 < numEls; c1++) {
			r = gen(rng);
			setCell(c1, r);
		}
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}

// GETTERS

// Find the normal of a matrix
double CUDAMatrix::getNorm(int n) {
	int c1, c2;
	double sum, max = 0;
	if (n == 1) {
		// 1 Norm
		for (c1 = 0; c1 < numCols; c1++) {
			sum = 0;
			for (c2 = 0; c2 < numRows; c2++) {
				sum += abs(getCell(c1, c2));
			}
			if (sum > max) {
				max = sum;
			}
		}
		return max;
	} else if (n == INFINITY) {
		// Inf Norm
		for (c1 = 0; c1 < numRows; c1++) {
			sum = 0;
			for (c2 = 0; c2 < numCols; c2++) {
				sum += abs(getCell(c1, c2));
			}
			if (sum > max) {
				max = sum;
			}
		}
		return max;
	} else {
		// Euclidian
		sum = 0;
		for (c1 = 0; c1 < numRows*numCols; c1++) {
			sum += std::pow(getCell(c1), n);
		}
		return std::pow(sum, 1.0 / n);
	}
}

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

// UTILS

int utils::getNumDigits(double x) {
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
// Find the maximum of 2 integers
int utils::max(int x, int y) {
	if (x > y) {
		return x;
	} else {
		return y;
	}
}
// Find the maximum of 2 doubles
double utils::max(double x, double y) {
	if (x > y) {
		return x;
	} else {
		return y;
	}
}
// Find the minimum of 2 integers
int utils::min(int x, int y) {
	if (x < y) {
		return x;
	} else {
		return y;
	}
}
// Find the minimum of 2 doubles
double utils::min(double x, double y) {
	if (x < y) {
		return x;
	} else {
		return y;
	}
}

// OPERATOR OVERRIDES

// <<
std::ostream& operator<<(std::ostream& oStream, CUDAMatrix& A) {
	if (A.isInitialised()) {
		double cell;
		int c1, c2, length, maxLength = 0, precision = 0;
		oStream << std::endl;
		for (c1 = 0; c1 < A.getNumEls(); c1++) {
			// Get precision
			cell = A.getCell(c1);
			if (cell - (int) cell > 0.0) {
				precision = 3;
			}
			// Get maximum number length
			length = utils::getNumDigits(cell);
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
			length = utils::getNumDigits(cell);
			for (c2 = 0; c2 < (maxLength - length); c2++) {
				oStream << " ";
			}
			// Output number
			oStream << std::setprecision(precision) << std::fixed << cell << " ";
			// Output new line if row end reached
			if (A.getCurRow(c1 + 1) > A.getCurRow(c1)) {
				oStream << "|";
				if (A.getCurRow(c1 + 1) < A.getNumRows()) {
					oStream << std::endl;
				}
			}
		}
		oStream << std::endl;
		return oStream;
	} else {
		throw std::runtime_error("Cannot perform matrix operations before initialisation");
	}
}
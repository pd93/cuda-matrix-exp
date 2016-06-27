# CUDAMatrixExp

This library was written as a part of my university dissertation. My report looks at the advantages of running complex matrix operations on a dedicated hardware device such as a graphics card (GPU) over a conventional processor (CPU). The focus of this library is a [Pade Approximation Algorithm](https://en.wikipedia.org/wiki/Pad%C3%A9_approximant) which is commonly used in Quantum Control to calculate the exponential of a matrix.

The library is written in **C++** and **CUDA** and runs on any Windows or Linux machine with a CUDA capable device (Compute Compatibility >= v2.0).

## Example:

Example.h
``` cpp
#include "src/CUDAMatrix.cuh"
int main(int argc, char **argv);
```

Example.cpp
``` cpp
#include "Main.h"

int main(int argc, char **argv) {

	try {

		// Set matrix size
		int size = 5;

		// Input variables
		std::complex<double> i = std::complex<double>(0, 1);
		CUDAMatrix A(size, {
			1, 0, 0, 0, 0,
			0, 2, 0, 0, 0,
			0, 0, 3, 0, 0,
			0, 0, 0, 4, 0,
			0, 0, 0, 0, 5
		});

        // Result variables
		CUDAMatrix eA(size);
		CUDAMatrix eAi(size);

		// Create timers
		CUDATimer t1, t2;

        // Calculations
		t1 = CUDAMatrix::exp(A, eA);
		t2 = CUDAMatrix::mul(eA, i, eAi);

		// Output
		std::cout << "A" << A << std::endl;
		std::cout << "e^A" << eA << t1 << std::endl;
		std::cout << "e^A * i" << eAi << t2 << std::endl;

	} catch (std::exception e) {
		std::cout << std::endl << e.what() << std::endl;
	}

	return 0;

}
```

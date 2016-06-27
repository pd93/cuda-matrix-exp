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

#include "Main.h"

int main(int argc, char **argv) {

	try {

		int size = 5;

		// Create variables
		std::complex<double> i = std::complex<double>(5, 3);
		CUDAMatrix A(size, {
			1, 0, 0, 0, 0,
			0, 2, 0, 0, 0,
			0, 0, 3, 0, 0,
			0, 0, 0, 4, 0,
			0, 0, 0, 0, std::complex<double>(0, 5)
		});
		CUDAMatrix eA(size);
		CUDAMatrix eAi(size);

		// Calculations
		CUDATimer t1, t2;
		t1 = CUDAMatrix::exp(A, eA);
		//t2 = CUDAMatrix::mul(eA, i, eAi);

		// Output
		std::cout << "A" << A << std::endl;
		std::cout << "e^A" << eA << t1 << std::endl;
		//std::cout << "e^A * i" << eAi << t2 << std::endl;
		
	} catch (std::exception e) {
		std::cout << std::endl << e.what() << std::endl;
	}

	return 0;

}
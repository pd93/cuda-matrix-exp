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

		CUDATimer t;

		CUDAMatrix A(5, {
			10, 2, 23, 13, 2,
			3, 5, 41, 18, 3,
			35, 12, 3, 13, 14,
			7, 22, 26, 2, 35,
			24, 31, 3, 66, 18
		});
		//std::cout << "A" << A;

		CUDAMatrix eA = CUDAMatrix(A.getNumRows(), A.getNumCols());
		t = CUDAMatrix::exp(A, eA);
		std::cout << "e^A" << eA << std::setprecision(5) << std::fixed << t.getTime()/1000 << "s";

	} catch (std::exception e) {
		std::cout << std::endl << e.what() << std::endl;
	}

	return 0;

}
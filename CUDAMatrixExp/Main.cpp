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

		//std::cout << "Create A" << std::endl;
		//CUDAMatrix A(5, {
		//	10, 2, 23, 13, 2,
		//	3, 5, 41, 18, 3,
		//	35, 12, 3, 13, 14,
		//	7, 22, 26, 2, 35,
		//	24, 31, 3, 66, 18
		//});
		//std::cout << A << std::endl;

		//std::cout << "Create B" << std::endl;
		//CUDAMatrix B(5, {
		//	2, 3, 0, 1, 1,
		//	3, 2, 1, 2, 4,
		//	2, 1, 3, 3, 2,
		//	0, 2, 1, 2, 3,
		//	0, 3, 1, 3, 1
		//});
		//std::cout << B << std::endl;

		std::cout << "Create A";
		CUDAMatrix A(3);
		A.setIdentity();
		std::cout << A;

		//std::cout << "Create R1 = add(A, B) " << std::endl;
		//CUDAMatrix R1 = CUDAMatrix(A.getNumRows(), A.getNumCols());
		//t = CUDAMatrix::add(A, B, R1);
		//std::cout << R1 << std::endl << std::setprecision(5) << std::fixed << t.getTime() << "s" << std::endl << std::endl;

		//std::cout << "Create R2 = sub(A, B) " << std::endl;
		//CUDAMatrix R2 = CUDAMatrix(A.getNumRows(), A.getNumCols());
		//t = CUDAMatrix::sub(A, B, R2);
		//std::cout << R2 << std::endl << std::setprecision(5) << std::fixed << t.getTime() << "s" << std::endl << std::endl;

		//std::cout << "Create R3 = mul(A, B) " << std::endl;
		//CUDAMatrix R3 = CUDAMatrix(A.getNumRows(), A.getNumCols());
		//t = CUDAMatrix::mul(A, B, R3);
		//std::cout << R3 << std::endl << std::setprecision(5) << std::fixed << t.getTime() << "s" << std::endl << std::endl;

		//std::cout << "Create R4 = inv(B) " << std::endl;
		//CUDAMatrix R4 = CUDAMatrix(A.getNumRows(), A.getNumCols());
		//CUDAMatrix::inv(B, R4);
		//std::cout << R4 << std::endl << std::setprecision(5) << std::fixed << t.getTime() << "s" << std::endl << std::endl;

		std::cout << "Create R5 = exp(A) " << std::endl;
		CUDAMatrix R5 = CUDAMatrix(A.getNumRows(), A.getNumCols());
		t = CUDAMatrix::exp(A, R5);
		std::cout << R5 << std::endl << std::setprecision(5) << std::fixed << t.getTime() << "s" << std::endl << std::endl;

	} catch (std::exception e) {
		std::cout << std::endl << e.what() << std::endl;
	}

	return 0;

}
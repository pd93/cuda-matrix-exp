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

	float time;

	std::cout << "Create A" << std::endl;
	CUDAMatrix A(5, {
		10, 2, 23, 13, 2,
		3, 5, 41, 18, 3,
		35, 12, 3, 13, 14,
		7, 22, 26, 2, 35,
		24, 31, 3, 66, 18
	});
	std::cout << A << std::endl;

	std::cout << "Create B" << std::endl;
	CUDAMatrix B(5, {
		2, 3, 0, 1, 1,
		3, 2, 1, 2, 4,
		2, 1, 3, 3, 2,
		0, 2, 1, 2, 3,
		0, 3, 1, 3, 1
	});
	std::cout << B << std::endl;

	std::cout << "Create R1 = add(A, B)" << std::endl;
	CUDAMatrix R1(A.getNumRows(), A.getNumCols());
	time = CUDAMatrix::add(A, B, R1);
	std::cout << R1;
	std::cout << std::setprecision(5) << std::fixed << time << "s" << std::endl << std::endl;

	std::cout << "Create R2 = sub(A, B)" << std::endl;
	CUDAMatrix R2(A.getNumRows(), A.getNumCols());
	time = CUDAMatrix::sub(A, B, R2);
	std::cout << R2;
	std::cout << std::setprecision(5) << std::fixed << time << "s" << std::endl << std::endl;

	std::cout << "Create R3 = mul(A, B)" << std::endl;
	CUDAMatrix R3(A.getNumRows(), A.getNumCols());
	time = CUDAMatrix::mul(A, B, R3);
	std::cout << R3;
	std::cout << std::setprecision(5) << std::fixed << time << "s" << std::endl << std::endl;

	std::cout << "Create R4 = inv(B)" << std::endl;
	CUDAMatrix R4(A.getNumRows(), A.getNumCols());
	CUDAMatrix R5(A.getNumCols(), A.getNumRows());
	CUDAMatrix R6(A.getNumRows(), A.getNumCols());
	time = CUDAMatrix::inv(B, R4);
	time += CUDAMatrix::tra(R4, R5);
	time += CUDAMatrix::mul(R4, R5, R6);
	std::cout << R4 << std::endl;
	std::cout << R5 << std::endl;
	std::cout << R6 << std::endl;
	std::cout << std::setprecision(5) << std::fixed << time << "s" << std::endl << std::endl;
	
	return 0;

}
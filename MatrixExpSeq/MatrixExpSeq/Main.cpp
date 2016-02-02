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

#include "Main.hpp"

int main(int argc, char **argv) {

	// Create A
	std::vector<std::vector<int>> vectorA({
		{ 1, 2, 3 },
		{ 4, 5, 6 }
	});
	Matrix matrixA(vectorA);
	printf("\nCreate A =\n");
	printf(matrixA.toString().c_str());

	// Create B
	std::vector<std::vector<int>> vectorB({
		{ 1, 2 },
		{ 4, 5 },
		{ 7, 8 }
	});
	Matrix matrixB(vectorB);
	printf("\nCreate B =:\n");
	printf(matrixB.toString().c_str());

	// Create C = A * B
	Matrix matrixC = matrixA.mul(matrixB);
	printf("\nCreate C = A * B =\n");
	printf(matrixC.toString().c_str());

	// Create D = C ^ 2
	//Matrix matrixD = matrixC.pow(2);
	printf("\nCreate D = C ^ 2 =\n");
	//printf(matrixD.toString().c_str());

}
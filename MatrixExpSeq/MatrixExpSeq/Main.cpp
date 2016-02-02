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

	try {
		// Create A
		printf("\nCreate A =\n");
		std::vector<std::vector<int>> vectorA({
			{ 1, 2, 3 },
			{ 3, 2, 1 },
			{ 2, 1, 3 }
		});
		Matrix matrixA(vectorA);
		printf(matrixA.toString().c_str());

		// Create B
		printf("\nCreate B =\n");
		std::vector<std::vector<int>> vectorB({
			{ 1, 2, 3 },
			{ 4, 5, 6 },
			{ 7, 8, 9 }
		});
		Matrix matrixB(vectorB);
		printf(matrixB.toString().c_str());

		// Create C = A+B
		printf("\nCreate C = A+B =\n");
		Matrix matrixC(matrixA.add(matrixB));
		printf(matrixC.toString().c_str());

		// Create D1 = A*3
		printf("\nCreate D1 = A*3 =\n");
		Matrix matrixD1(matrixA.mul(3));
		printf(matrixD1.toString().c_str());

		// Create D2 = A*B
		printf("\nCreate D2 = A*B =\n");
		Matrix matrixD2(matrixA.mul(matrixB));
		printf(matrixD2.toString().c_str());

		// Create E = B^2
		printf("\nCreate E = B^2 =\n");
		Matrix matrixE(matrixB.pow(2));
		printf(matrixE.toString().c_str());

		// Create F = e^A =
		printf("\nCreate F = e^A =\n");
		Matrix matrixF(matrixA.exp(6));
		printf(matrixF.toString().c_str());
	}
	catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

}
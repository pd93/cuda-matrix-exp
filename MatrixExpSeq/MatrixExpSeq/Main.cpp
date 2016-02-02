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
		std::vector<std::vector<double>> vectorA({
			{ 1, 2, 3 },
			{ 3, 2, 1 },
			{ 2, 1, 3 }
		});
		Matrix matrixA(vectorA);
		matrixA.printm();

		// Create B
		printf("\nCreate B =\n");
		std::vector<std::vector<double>> vectorB({
			{ 1, 2, 3 },
			{ 4, 5, 6 },
			{ 7, 8, 9 }
		});
		Matrix matrixB(vectorB);
		matrixB.printm();

		// Create C = A+B
		printf("\nCreate C = A+B =\n");
		Matrix matrixC(matrixA.add(matrixB));
		matrixC.printm();

		// Create D1 = A*3
		printf("\nCreate D1 = A*3 =\n");
		Matrix matrixD1(matrixA.mul(3));
		matrixD1.printm();

		// Create D2 = A*B
		printf("\nCreate D2 = A*B =\n");
		Matrix matrixD2(matrixA.mul(matrixB));
		matrixD2.printm();

		// Create E = B^2
		printf("\nCreate E = B^2 =\n");
		Matrix matrixE(matrixB.pow(2));
		matrixE.printm();

		// Create F = e^A =
		printf("\nCreate F = e^A =\n");
		Matrix matrixF(matrixA.exp(50));
		matrixF.printm();
	}
	catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

}
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

		clock_t t;

		// Create A
		printf("\nCreate A =\n");
		std::vector<std::vector<double>> vectorA({
			{ 1, 2, 3 },
			{ 3, 2, 1 },
			{ 2, 1, 3 }
		});
		Matrix matrixA(vectorA);
		Matrix* A = &matrixA;
		matrixA.printm();

		// Create B
		printf("\nCreate B =\n");
		std::vector<std::vector<double>> vectorB({
			{ 1, 2, 3 },
			{ 4, 5, 6 },
			{ 7, 8, 9 }
		});
		Matrix matrixB(vectorB);
		Matrix* B = &matrixB;
		matrixB.printm();

		// Create I (Identity)
		printf("\nCreate I (Identity) =\n");
		Matrix matrixI(3, 3);
		Matrix* I = &matrixI;
		matrixI.setIdentity();
		matrixI.printm();

		// Create C = A+B
		printf("\nCreate C = A+B =\n");
		t = clock();
		Matrix matrixC;
		Matrix C(matrixA.add(matrixB));
		t = clock() - t;
		matrixC.printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		//// Create D1 = A*3
		//printf("\nCreate D1 = A*3 =\n");
		//t = clock();
		//Matrix matrixD1(matrixA.mul(3));
		//t = clock() - t;
		//matrixD1.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		//// Create D2 = A*B
		//printf("\nCreate D2 = A*B =\n");
		//t = clock();
		//Matrix matrixD2(matrixA.mul(matrixB));
		//t = clock() - t;
		//matrixD2.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		//// Create E = B^2
		//printf("\nCreate E = B^2 =\n");
		//t = clock();
		//Matrix matrixE(matrixB.pow(2));
		//t = clock() - t;
		//matrixE.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		//// Create F = e^A (Taylor) =
		//printf("\nCreate F = e^A (Taylor) =\n");
		//t = clock();
		//Matrix matrixF(matrixA.mExp(30, 't'));
		//t = clock() - t;
		//matrixF.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		//// Create G = e^A (Pade) =
		//printf("\nCreate G = e^A (Pade) =\n");
		//t = clock();
		//Matrix matrixG(matrixA.mExp(10, 'p'));
		//t = clock() - t;
		//matrixG.printm();
		//printf("%.9f seconds\n", (float) (t)/CLOCKS_PER_SEC);

		//// Create H


		//// Create J = e^I (Diagonal)
		//printf("\nCreate J = e^I (Diagonal) =\n");
		//t = clock();
		//Matrix matrixJ(matrixI.mExp(30));
		//t = clock() - t;
		//matrixJ.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);
	}
	catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

}
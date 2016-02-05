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

using namespace std;

int main(int argc, char **argv) {

	try {

		clock_t t;

		// Create A
		cout << "\nCreate A =\n" << endl;
		Matrix* A = new Matrix({
			{ 1, 2, 3 },
			{ 3, 2, 1 },
			{ 2, 1, 3 }
		});
		A->printm();

		// Create B
		cout << "\nCreate B =\n";
		Matrix* B = new Matrix({
			{ 1, 2, 3 },
			{ 4, 5, 6 },
			{ 7, 8, 9 }
		});
		B->printm();

		// Create I (Identity)
		cout << "\nCreate I (Identity) =\n";
		Matrix* I = new Matrix(3, 3);
		I->setIdentity();
		I->printm();

		// Create C = A+B
		printf("\nCreate C = A+B =\n");
		t = clock();
		Matrix* C = Matrix::add(A, B);
		t = clock() - t;
		C->printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Create D1 = A*3
		printf("\nCreate D1 = A*3 =\n");
		t = clock();
		Matrix* D1 = Matrix::mul(A, 3);
		t = clock() - t;
		D1->printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Create D2 = A*B
		printf("\nCreate D2 = A*B =\n");
		t = clock();
		Matrix* D2 = Matrix::mul(A, B);
		t = clock() - t;
		D2->printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Create E = B^2
		printf("\nCreate E = B^2 =\n");
		t = clock();
		Matrix* E = Matrix::pow(B, 2);
		t = clock() - t;
		E->printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Create F = e^A (Taylor) =
		printf("\nCreate F = e^A (Taylor) =\n");
		t = clock();
		Matrix* F = Matrix::mExp(A, 't', 20);
		t = clock() - t;
		F->printm();
		printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Create G = e^A (Pade) =
		printf("\nCreate G = e^A (Pade) =\n");
		t = clock();
		Matrix* G = Matrix::mExp(A, 'p', 10);
		t = clock() - t;
		G->printm();
		printf("%.9f seconds\n", (float) (t)/CLOCKS_PER_SEC);

		//// Create H


		//// Create J = e^I (Diagonal)
		//printf("\nCreate J = e^I (Diagonal) =\n");
		//t = clock();
		//Matrix matrixJ(matrixI.mExp(30));
		//t = clock() - t;
		//matrixJ.printm();
		//printf("%.9f seconds\n", (float)(t) / CLOCKS_PER_SEC);

		// Delete arrays and 
		delete A, B, C, I;
		A, B, C, I = nullptr;
	}
	catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

	return 0;

}
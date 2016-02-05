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

		chrono::high_resolution_clock::time_point start;
		chrono::high_resolution_clock::time_point end;
		double duration;

		cout << endl << "############################# Create Matrices #############################" << endl;

		// Create A
		cout << endl << "Create A =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* A = new Matrix({
			{ 1, 2, 3, 3, 1 },
			{ 3, 2, 1, 2, 3 },
			{ 1, 2, 3, 3, 1 },
			{ 3, 2, 1 ,2, 3 },
			{ 2, 1, 3 ,2, 1 }
		});
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << A;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create I (Identity)
		cout << endl << "Create I (Identity) =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* I = new Matrix(5, 5);
		I->setIdentity();
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << I;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create R (Random 0-9)
		cout << endl << "Create R (Random 0-9) =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* R = new Matrix(5, 5);
		R->setRandom(0, 9);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << R;
		cout << setprecision(5) << duration << " seconds" << endl;

		cout << endl << "############################# Basic Functions #############################" << endl;

		// Create B = A+R
		cout << endl << "Create B = A+R =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* B = Matrix::add(A, R);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << B;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create C = A*3
		cout << endl << "Create C = A*3" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* C = Matrix::mul(A, 3);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << C;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create D = A*R
		cout << endl << "Create D = A*R =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* D = Matrix::mul(A, I);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << D;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create E = A^2
		cout << endl << "Create E = A^2 =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* E = Matrix::pow(A, 2);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << E;
		cout << setprecision(5) << duration << " seconds" << endl;

		cout << endl << "############################# Matrix Exponentials #############################" << endl;

		// Create F = e^A (Taylor) =
		cout << endl << "Create F = e^A (Taylor) =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* F = Matrix::mExp(A, 't', 50);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << F;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create G = e^A (Pade) =
		cout << endl << "Create G = e^A (Pade) =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* G = Matrix::mExp(A, 'p', 10);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << G;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Create H = e^I (Diagonal)
		cout << endl << "Create H = e^I (Diagonal) =" << endl;
		start = chrono::high_resolution_clock::now();
		Matrix* H = Matrix::mExp(I);
		end = chrono::high_resolution_clock::now();
		duration = (chrono::duration_cast<chrono::microseconds>(end - start).count() / 100000.0);
		cout << H;
		cout << setprecision(5) << duration << " seconds" << endl;

		// Delete arrays and set to nullptr
		delete A, B, C, D, E, F, G, H, I, R;
		A, B, C, D, E, F, G, H, I, R = nullptr;
	} catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

	return 0;

}
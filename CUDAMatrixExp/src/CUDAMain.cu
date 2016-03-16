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

#include "CUDAMain.cuh"

using namespace std;

int main(int argc, char **argv) {

	try {

		CUDAMatrix A(5, 5, {
			1, 2, 3, 3, 1,
			3, 2, 1, 2, 3,
			1, 2, 3, 3, 1,
			3, 2, 1, 2, 3,
			2, 1, 3, 2, 1
		});

		CUDAMatrix B(1, 2, {
			2, 3, 0, 1, 1,
			3, 2, 1, 2, 4,
			2, 1, 3, 3, 2,
			0, 2, 1, 2, 3,
			0, 3, 1, 3, 1
		});

		CUDAMatrix R(5, 5);

		CUDAMatrix::mul(A, B, R);

		cout << R;

	} catch (int e) {
		printf("\n||||||||||||||||\n|| Error: %i ||\n||||||||||||||||\n", e);
	}

	return 0;

}
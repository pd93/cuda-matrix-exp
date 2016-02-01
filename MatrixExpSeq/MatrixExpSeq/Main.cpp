#include "Main.hpp"

int main(int argc, char **argv) {

	std::vector<std::vector<int>> vectorA({
		{ 1, 2, 3 },
		{ 4, 5, 6 }
	});
	std::vector<std::vector<int>> vectorB({
		{ 6, 5 },
		{ 4, 3 },
		{ 2, 1 }
	});
	Matrix matrixA(vectorA);
	Matrix matrixB(2, 2);
	Matrix matrixC = matrixA.mul(matrixB);
	//
	// Print results and pause
	printf("Input Matricies:\n");
	printf(matrixA.toString().c_str());
	printf(matrixB.toString().c_str());
	printf("Output Matrix:\n");
	printf(matrixC.toString().c_str());
	getchar(); // Pause

}
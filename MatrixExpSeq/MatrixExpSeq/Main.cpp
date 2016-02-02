#include "Main.hpp"

int main(int argc, char **argv) {

	std::vector<std::vector<int>> vectorA({
		{ 1, 2, 3 },
		{ 4, 5, 6 }
	});
	std::vector<std::vector<int>> vectorB({
		{ 6, 5, 5, 42 },
		{ 4, 3, 5, 7 },
		{ 2, 1, 5, 4 }
	});
	Matrix matrixA(vectorA);
	Matrix matrixB(vectorB);
	Matrix matrixC = matrixA.mul(matrixB);
	
	// Print results and pause
	printf("Matrix A:\n");
	printf(matrixA.toString().c_str());
	printf("Matrix B:\n");
	printf(matrixB.toString().c_str());
	printf("Matrix A * Matrix B:\n");
	printf(matrixC.toString().c_str());
	getchar(); // Pause

}
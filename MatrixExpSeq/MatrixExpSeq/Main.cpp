#include "Main.hpp"

int main(int argc, char **argv) {

	std::vector<std::vector<int>> vectorA({
		{ 1, 2, 3 },
		{ 4, 5, 6 }
	});
	std::vector<std::vector<int>> vectorB({
		{ 1, 2, 3 },
		{ 4, 5, 6 },
		{ 7, 8, 9 }
	});
	Matrix matrixA(vectorA);
	Matrix matrixB(vectorB);
	Matrix matrixC = matrixA.mul(matrixB);
	Matrix matrixD = matrixC.pow(2);
	
	// Print results and pause
	printf("Matrix A:\n");
	printf(matrixA.toString().c_str());
	printf("Matrix B:\n");
	printf(matrixB.toString().c_str());
	printf("Matrix A * Matrix B:\n");
	printf(matrixC.toString().c_str());
	printf("(Matrix A * Matrix B) ^ 2:\n");
	printf(matrixD.toString().c_str());
	getchar(); // Pause

}
//
// Author: Pete Davison (2016)
//

#ifndef matrix_h
#define matrix_h
#include <iostream>
#include <string>
#include <vector>

class Matrix {
private:
	std::vector<std::vector<int>> matrix;
	int numRows, numCols;
	bool initialised;
public:
	// Constructors
	Matrix();
	Matrix(int inNumRows, int inNumCols);
	Matrix(std::vector<std::vector<int>> inMatrix);
	void init(int inNumRows, int inNumCols);
	// Matrix Functions
	Matrix mul(Matrix matrixB);
	Matrix pow(int power);
	Matrix exp(int order);
	void setZero();
	void setIdentity();
	// Output
	std::string toString();
	// Getters
	int getNumRows();
	int getNumCols();
	// Setters
	void setMatrix(std::vector<std::vector<int>> inMatrix);
	void setCell(int row, int col, int value);
	// Booleans
	bool isSquare();
	bool isDiagonal();
	bool isScalar();
	bool isIdentity();
	bool isZero();
};

#endif
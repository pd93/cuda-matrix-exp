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

#ifndef matrix_h
#define matrix_h
#include <iostream>
#include <string>
#include <vector>

class Matrix {
private:
	std::vector<std::vector<double>> matrix;
	int numRows, numCols;
	bool initialised;
public:
	// Constructors
	Matrix();
	Matrix(int inNumRows, int inNumCols);
	Matrix(std::vector<std::vector<double>> inMatrix);
	Matrix(const Matrix &obj);
	void init(int inNumRows, int inNumCols);
	// Matrix Functions
	Matrix add(Matrix martixB);
	Matrix mul(Matrix matrixB);
	Matrix mul(double m);
	Matrix pow(int p);
	Matrix exp(int n);
	void setZero();
	void setIdentity();
	// Booleans
	bool isSquare();
	bool isDiagonal();
	bool isScalar();
	bool isIdentity();
	bool isZero();
	// Getters
	int getNumRows();
	int getNumCols();
	// Setters
	void setMatrix(std::vector<std::vector<double>> inMatrix);
	void setCell(int row, int col, double value);
	// Output
	void printm();
};

#endif
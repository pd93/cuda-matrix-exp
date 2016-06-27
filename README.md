# CUDAMatrixExp

This project looks at the advantages of running complex matrix operations on a dedicated hardware device such as a graphics card (GPU) over the conventional method of running it on the CPU. The focus of the program is a Pade Approximation which is commonly used in Quantum Control to calculate the exponential of a matrix.

Matrix exponentials are vital to understanding how a particle's energy state change over time. These calculations can get very intensive when the problems becomes large and this report investigates whether dedicated devices can help to solve this problem.

The program produced is written in C++ and CUDA and runs on any Windows or Linux machine with a CUDA capable device (Compute Compatibility >= v2.0).

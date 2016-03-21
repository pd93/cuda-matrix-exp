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

// Include header file
#include "Timer.cuh"

Timer::Timer() {
	cudaEventCreate(start);
	cudaEventCreate(stop);
}
Timer::~Timer() {
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
void Timer::start() {
	cudaEventRecord(start, 0);
}
void Timer::stop() {
	cudaEventRecord(stop, 0);
}
void Timer::print() {
	cudaEventElapsedTime(&time, start, stop);
	std::cout << time;
}
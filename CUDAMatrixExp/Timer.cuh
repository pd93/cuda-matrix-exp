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

// Precompiler include check
#ifndef cudamatrix_h
#define cudamatrix_h
// Include CUDA Stuff
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Timer {
private:
	float time;
	cudaEvent_t start, stop;
public:
	Timer();
	~Timer();
	void start();
	void stop();
	void print();
};

#endif
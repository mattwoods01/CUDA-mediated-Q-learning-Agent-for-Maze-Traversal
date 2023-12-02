/*************************************************************************
/* ECE 277: GPU Programmming 2020 
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_madd(int* A, int* B, int* C, int M, int N);

void cu_madd(int* A, int* B, int* C, int M, int N)
{
	int *d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(unsigned int)*M*N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_madd << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_madd(int* A, int* B, int* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;

	if (idx == 0)
		printf("cuda matrix (%d, %d) addition\n", N, M);

	if (ix < M && iy < N)
		C[idx] = A[idx] + B[idx];
}


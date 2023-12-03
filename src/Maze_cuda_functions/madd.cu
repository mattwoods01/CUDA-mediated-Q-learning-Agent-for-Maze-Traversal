/*************************************************************************
/* ECE 277: GPU Programmming 2020 
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_madd(int* A, int* B, int* C, int M, int N);
__global__ void epsilonGreedyKernel(float* exploration_rates, int current_episode, int num_episodes, float exploration_start, float exploration_end);

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

__global__
void epsilonGreedyKernel(float* exploration_rates, int current_episode, int num_episodes, float exploration_start, float exploration_end) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0) {
		exploration_rates[0] = exploration_start * pow(exploration_end / exploration_start, static_cast<float>(current_episode) / static_cast<float>(num_episodes));
	}
}

void epsilonGreedyCUDA(float* exploration_rates, int current_episode, int num_episodes, float exploration_start, float exploration_end) {
	float* d_exploration_rates;

	int dimx = 32;
	dim3 block(dimx, 1);
	dim3 grid((4 + block.x - 1) / block.x, 1);

	cudaMalloc((void**)&d_exploration_rates, sizeof(float));
	cudaMemcpy(d_exploration_rates, exploration_rates, sizeof(float), cudaMemcpyHostToDevice);
	epsilonGreedyKernel << <32, block >> > (d_exploration_rates, current_episode, num_episodes, exploration_start, exploration_end);
	cudaMemcpy(exploration_rates, d_exploration_rates, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_exploration_rates);
	cudaDeviceSynchronize();
}




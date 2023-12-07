#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <queue>
#include <iostream>

__global__ void epsilonGreedyKernel(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
__global__ void randomArrayKernel(int* maze_array, int height, int width, unsigned long long seed);
__global__ void randomizeZerosKernel(int* array, int size, float percentage, unsigned long long seed);

__global__ void epsilonGreedyKernel(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_episodes) {
        float frac = static_cast<float>(tid) / static_cast<float>(num_episodes);
        exploration_rates[tid] = exploration_start * exp(frac * log(exploration_end / exploration_start));
    }
}

void epsilonGreedyCUDA(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end) {
    float* d_exploration_rates;

    cudaMalloc((void**)&d_exploration_rates, num_episodes * sizeof(float));
    cudaMemcpy(d_exploration_rates, exploration_rates, num_episodes * sizeof(float), cudaMemcpyHostToDevice);

    int dimx = 32;
    int blocks_per_grid = (num_episodes + dimx - 1) / dimx;
    dim3 block(dimx, 1);
    dim3 grid(blocks_per_grid, 1);

    epsilonGreedyKernel << <grid, block >> > (d_exploration_rates, num_episodes, exploration_start, exploration_end);

    cudaMemcpy(exploration_rates, d_exploration_rates, num_episodes * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_exploration_rates);
    cudaDeviceSynchronize();
}

__device__ void dfs(int* maze_array, int height, int width, int idx_x, int idx_y, curandState* state) {
    // Mark the current cell as visited
    int idx = idx_y * width + idx_x;
    maze_array[idx] = 3;  // Mark as visited (you can choose any marker)

    // Define the possible moves (up, down, left, right)
    int moves[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

    // Shuffle the moves array using Fisher-Yates algorithm
    for (int i = 4; i > 0; --i) {
        int j = curand_uniform(state) * (i + 1);
        // Swap moves[i] and moves[j]
        int temp0 = moves[i][0];
        int temp1 = moves[i][1];
        moves[i][0] = moves[j][0];
        moves[i][1] = moves[j][1];
        moves[j][0] = temp0;
        moves[j][1] = temp1;
    }

    // Explore neighbors in a random order
    for (int i = 0; i < 4; ++i) {
        int new_x = idx_x + moves[i][0];
        int new_y = idx_y + moves[i][1];

        // Check if the new position is within bounds and has not been visited
        if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
            int new_idx = new_y * width + new_x;
            if (maze_array[new_idx] == 0) {
                dfs(maze_array, height, width, new_x, new_y, state);
            }
        }
    }
}

__global__ void randomArrayKernel(int* maze_array, int height, int width, unsigned long long seed) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = idx_y * width + idx_x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Set the maze value randomly
    maze_array[idx] = curand_uniform(&state) < 0.55 ? 0 : 1;

    // Ensure that the values surrounding the first and last indices are 0
    if ((idx_x == 0 || idx_x == width - 1) && (idx_y == 0 || idx_y == height - 1)) {
        // Set neighboring values to 0
        maze_array[idx] = 0;
        maze_array[idx + 1] = 0;  // Right neighbor
        maze_array[idx - 1] = 0;  // Left neighbor
        maze_array[idx + width] = 0;  // Bottom neighbor
        maze_array[idx - width] = 0;  // Top neighbor
    }

    // Ensure connectivity by applying DFS from the top-left corner
    if (idx_x == 0 && idx_y == 0) {
        dfs(maze_array, height, width, idx_x, idx_y);  // Start DFS from the top-left corner
    }
}

void randomArrayCuda(int* maze_array, int height, int width, unsigned long long seed) {
    int* d_maze_array;

    cudaMalloc((void**)&d_maze_array, height * width * sizeof(float));
    cudaMemcpy(d_maze_array, maze_array, height * width * sizeof(float), cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    randomArrayKernel << <grid, block >> > (d_maze_array, height, width, seed);

    cudaMemcpy(maze_array, d_maze_array, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_maze_array);
    cudaDeviceSynchronize();
}

__global__ void randomizeZerosKernel(int* A, int size, float percentage, unsigned long long seed) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = idx_y * size + idx_x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    if (A[idx] == 1 && curand_uniform(&state) < percentage) {
        A[idx] = 2;
    }
}

void randomizeZerosCuda(int* A, int X, int Y, float percentage, unsigned long long seed) {
    int* d_A;

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((X + block.x - 1) / block.x, (Y + block.y - 1) / block.y);

    int size = X * Y * sizeof(int);  

    cudaMalloc((void**)&d_A, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    randomizeZerosKernel << <grid, block >> > (d_A, X, percentage, seed);  

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaDeviceSynchronize();
}


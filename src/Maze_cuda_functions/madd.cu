#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void epsilonGreedyKernel(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
__global__ void randomArrayKernel(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed);
__global__ void randomizeZerosKernel(int* array, int size, float percentage, unsigned long long seed);
__global__ void dfs_kernel(int* maze_array, int width, int height, int start_x, int start_y, int end_x, int end_y, unsigned long long seed);
__global__ void guaranteePathKernel(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed);

__global__ void randomArrayKernel_ctrl(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed);
__global__ void epsilonGreedyKernel_ctrl(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
//__global__ void dfs_kernel_ctrl(int* maze_array, int width, int height, int start_x, int start_y, int end_x, int end_y, unsigned long long seed);

__global__ void epsilonGreedyKernel(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_episodes) {
        float frac = static_cast<float>(tid) / static_cast<float>(num_episodes);
        exploration_rates[tid] = exploration_start * exp(frac * log(exploration_end / exploration_start));
    }
}

void epsilonGreedyCUDA(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end, cudaStream_t stream = 0) {
    float* d_exploration_rates;

    cudaMalloc((void**)&d_exploration_rates, num_episodes * sizeof(float));

    // Use cudaMemcpyAsync with specified stream
    cudaMemcpyAsync(d_exploration_rates, exploration_rates, num_episodes * sizeof(float), cudaMemcpyHostToDevice, stream);

    int dimx = 32;
    int blocks_per_grid = (num_episodes + dimx - 1) / dimx;
    dim3 block(dimx, 1);
    dim3 grid(blocks_per_grid, 1);

    // Launch kernel with specified stream
    epsilonGreedyKernel << <grid, block, 0, stream >> > (d_exploration_rates, num_episodes, exploration_start, exploration_end);

    // Use cudaMemcpyAsync with specified stream
    cudaMemcpyAsync(exploration_rates, d_exploration_rates, num_episodes * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFree(d_exploration_rates);
}

__global__ void randomArrayKernel(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = idx_y * width + idx_x;

    __shared__ int shared_start_x, shared_start_y, shared_end_x, shared_end_y;

    // Only one thread in the block initializes shared variables
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_start_x = start_x;
        shared_start_y = start_y;
        shared_end_x = end_x;
        shared_end_y = end_y;
    }

    // Synchronize to make sure shared variables are initialized before use
    __syncthreads();

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Use shared_start_x, shared_start_y, shared_end_x, shared_end_y instead of start_x, start_y, end_x, end_y
    maze_array[idx] = curand_uniform(&state) < 0.3 ? 0 : 1;
    // Avoid global memory access in the loop
    //if (idx_x == shared_start_x && idx_y == shared_start_y)
    maze_array[shared_start_y * width + shared_start_x] = 2;

    //if (idx_x == shared_end_x && idx_y == shared_end_y)
    maze_array[shared_end_y * width + shared_end_x] = 3;
}

void randomArrayCuda(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int* d_maze_array;

    cudaMalloc((void**)&d_maze_array, height * width * sizeof(float));
    cudaMemcpy(d_maze_array, maze_array, height * width * sizeof(float), cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    randomArrayKernel << <grid, block >> > (d_maze_array, height, width, start_x, start_y, end_x, end_y, seed);

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
        A[idx] = 5;
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

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__ void dfs_kernel(int* maze_array, int width, int height, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int current_x = blockIdx.x * blockDim.x + threadIdx.x;
    int current_y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = current_y * width + current_x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Check if the current cell is the end cell
    if (current_x == end_x && current_y == end_y) {
        // You've reached the end, you can handle it as needed
        return;
    }

    // Define the possible moves (right, left, down, up)
    int moves[4][2] = { {0, height / 3}, {0, -height / 3}, {width / 3, 0}, {-width / 3, 0} };

    // Fisher-Yates shuffle to traverse randomly
    for (int i = 3; i > 0; --i) {
        int j = curand_uniform(&state) * (i + 1);

        // Swap moves[i] with moves[j]
        swap(moves[i][0], moves[j][0]);
        swap(moves[i][1], moves[j][1]);
    }

    // Check each possible move
    for (int i = 0; i < 4; ++i) {
        int new_x = current_x + moves[i][0];
        int new_y = current_y + moves[i][1];

        // Check if the new position is within bounds
        if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
            int new_idx = new_y * width + new_x;


            // Check if the new cell is open and not visited
            if (maze_array[new_idx] == 1) {
                if (curand_uniform(&state) < 0.1) {
                    maze_array[new_idx] = 0;
                }
                // Recursively call DFS on the new cell
                dfs_kernel << < 1, 1 >> > (maze_array, width, height, start_x, start_y, end_x, end_y, seed);

                // If the end has been reached in the recursive call, exit the loop
                if (maze_array[end_y * width + end_x] == 4) {
                    return;
                }
            }
        }
    }

    // Additional condition to prevent changing the starting cell

}

__global__ void guaranteePathKernel(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_idx = start_y * width + start_x;
    int end_idx = end_y * width + end_x;

    curandState_t state;
    curand_init(seed, idx_x + idx_y * width, 0, &state);

    // Set cells in the same row or column as start or end to 0 for only half of the width or height
    if (idx_y == start_y || idx_y == end_y) {
        int start_col = (idx_y == start_y) ? 0 : width / 2;
        int end_col = (idx_y == end_y) ? width : width / 2;
        for (int i = start_col; i < end_col; ++i) {
            maze_array[idx_y * width + i] = 0;
        }
    }

    if (idx_x == start_x || idx_x == end_x) {
        int start_row = (idx_x == start_x) ? 0 : height / 2;
        int end_row = (idx_x == end_x) ? height : height / 2;
        for (int i = start_row; i < end_row; ++i) {
            maze_array[i * width + idx_x] = 0;
        }
    }

    maze_array[start_idx] = 3;
    maze_array[end_idx] = 4;

    // Randomly select two additional spots and apply the same logic using curand
    if (curand_uniform(&state) < 0.005) {
        int rand_x1 = curand(&state) % width;
        int rand_y1 = curand(&state) % height;
        for (int i = 0; i < width / 2; ++i) {
            maze_array[rand_y1 * width + i] = 0;
        }

        int rand_x2 = curand(&state) % width;
        int rand_y2 = curand(&state) % height;
        for (int i = width / 2; i < width; ++i) {
            maze_array[rand_y2 * width + i] = 0;
        }
    }
}

void guranteePathCuda(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int* d_maze_array;

    cudaMalloc((void**)&d_maze_array, height * width * sizeof(int));
    cudaMemcpy(d_maze_array, maze_array, height * width * sizeof(int), cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    guaranteePathKernel << <grid, block >> > (d_maze_array, height, width, start_x, start_y, end_x, end_y, seed);

    cudaMemcpy(maze_array, d_maze_array, height * width * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_maze_array);
    cudaDeviceSynchronize();
}

///////////////////////////Control functions

//epsilonGreedykernel_non_async
__global__ void epsilonGreedyKernel_ctrl(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < num_episodes) {
    float frac = static_cast<float>(tid) / static_cast<float>(num_episodes);
    exploration_rates[tid] = exploration_start * exp(frac * log(exploration_end / exploration_start));
}
}

void epsilonGreedyCUDA_ctrl(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end) {
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

//randomArrayKernel_non_shared
__global__ void randomArrayKernel_ctrl(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

int idx = idx_y * width + idx_x;

curandState state;
curand_init(seed, idx, 0, &state);

// Set the maze value randomly
maze_array[idx] = curand_uniform(&state) < 0.4 ? 0 : 1;
maze_array[start_y * width + start_x] = 2;
maze_array[end_y * width + end_x] = 3;

}

void randomArrayCuda_ctrl(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int* d_maze_array;

    cudaMalloc((void**)&d_maze_array, height * width * sizeof(float));
    cudaMemcpy(d_maze_array, maze_array, height * width * sizeof(float), cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    randomArrayKernel << <grid, block >> > (d_maze_array, height, width, start_x, start_y, end_x, end_y, seed);

    cudaMemcpy(maze_array, d_maze_array, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_maze_array);
    cudaDeviceSynchronize();
}


void dfsCuda(int* maze_array, int height, int width, int start_x, int start_y, int end_x, int end_y, unsigned long long seed) {
    int* d_maze_array;
    cudaMalloc((void**)&d_maze_array, height * width * sizeof(int));
    cudaMemcpy(d_maze_array, maze_array, height * width * sizeof(int), cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    dfs_kernel << <grid, block >> > (d_maze_array, width, height, start_x, start_y, end_x, end_y, seed);

    cudaMemcpy(maze_array, d_maze_array, height * width * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_maze_array);
    cudaDeviceSynchronize();
}
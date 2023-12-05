#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void epsilonGreedyKernel(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
__global__ void randomArrayKernel(int* maze_array, int height, int width, unsigned long long seed);
__global__ void update_q_table_kernel(float* q_value, float next_q_value, int state_x, int state_y, int action, int next_state_x, int next_state_y, float reward, float learning_rate, float discount_factor);
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

__device__ void dfs(int* maze_array, int height, int width, int x, int y, curandState* state) {
    // Mark the current cell as visited
    maze_array[y * width + x] = 0;

    // Define possible directions (up, down, left, right)
    int directions[4][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} };

    // Shuffle the directions randomly
    for (int i = 0; i < 4; ++i) {
        int rand_idx = static_cast<int>(curand_uniform(state) * 2);
        int temp_x = x + directions[rand_idx][0];
        int temp_y = y + directions[rand_idx][1];

        // Check if the new cell is within bounds
        if (temp_x >= 0 && temp_x < width && temp_y >= 0 && temp_y < height) {
            // Check if the neighboring cell is visited
            if (maze_array[temp_y * width + temp_x] == 1) {
                // Recursively call dfs for the adjacent cell
                maze_array[temp_y * width + temp_x] = 0;
                maze_array[(y + temp_y) / 2 * width + (x + temp_x) / 2] = 0;
                dfs(maze_array, height, width, temp_x, temp_y, state);
                
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
        dfs(maze_array, height, width, idx_x, idx_y, &state);
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

__global__ void update_q_table_kernel(float* q_value, float next_q_value, int state_x, int state_y, int action, int next_state_x, int next_state_y, float reward, float learning_rate, float discount_factor) {
    // Calculate the index directly, assuming q_value is a single value

    *q_value = *q_value + learning_rate * (reward + discount_factor * next_q_value - *q_value);
}

float* d_q_value = nullptr;

void update_q_table_cuda(float* q_value, float next_q_value, int state_x, int state_y, int action, int next_state_x, int next_state_y, float reward, float learning_rate, float discount_factor) {
    // Check if device memory is allocated, if not, allocate it
    if (!d_q_value) {
        cudaMalloc((void**)&d_q_value, sizeof(float));
    }

    // Copy the input q_value to the device
    cudaMemcpy(d_q_value, q_value, sizeof(float), cudaMemcpyHostToDevice);

    // Use a 1D grid and block for simplicity, as you are dealing with a single value
    dim3 block(1);
    dim3 grid(1);

    // Update q_value on the device
    update_q_table_kernel << <grid, block >> > (d_q_value, next_q_value, state_x, state_y, action, next_state_x, next_state_y, reward, learning_rate, discount_factor);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(q_value, d_q_value, sizeof(float), cudaMemcpyDeviceToHost);
}

void cleanup_cuda() {
    if (d_q_value != nullptr) {
        cudaFree(d_q_value);
        d_q_value = nullptr;
    }
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


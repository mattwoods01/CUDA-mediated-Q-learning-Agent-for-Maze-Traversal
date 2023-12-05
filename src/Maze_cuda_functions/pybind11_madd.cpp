#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>

namespace py = pybind11;

extern void epsilonGreedyCUDA(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
extern void randomArrayCuda(int* maze_arrayu, int height, int width, unsigned long long seed);
extern void update_q_table_cuda(float* q_value, float next_q_value, int state_x, int state_y, int action, int next_state_x, int next_state_y, float reward, float learning_rate, float discount_factor);
extern void randomizeZerosCuda(int* A, int X, int Y, float percentage, unsigned long long seed);


py::array_t<float> py_epsilonGreedyCUDA(int num_episodes, float exploration_start, float exploration_end) {
    // Create a NumPy array to hold the results
    py::array_t<float> result_array(num_episodes);
    py::buffer_info buf_info = result_array.request();
    float* ptr = static_cast<float*>(buf_info.ptr);

    // Call the CUDA function
    epsilonGreedyCUDA(ptr, num_episodes, exploration_start, exploration_end);

    return result_array;
}

py::array_t<int> randomArrayWrapper(int height, int width, unsigned long long seed) {
    // Create a NumPy array to hold the results
    py::array_t<int> result_array({ height, width });
    py::buffer_info buf_info = result_array.request();
    int* ptr = static_cast<int*>(buf_info.ptr);

    // Call the CUDA function
    randomArrayCuda(ptr, height, width, seed);

    return result_array;
}

py::array_t<float> update_q_table_gpu(float q_value, float next_q_value, int state_x, int state_y, int action, int next_state_x, int next_state_y, float reward, float learning_rate, float discount_factor) {
    // Create a numpy array to hold the result
    py::array_t<float> result_array({ 1 });
    py::buffer_info buf_array = result_array.request();
    float* ptr = static_cast<float*>(buf_array.ptr);

    // Call the CUDA kernel
    update_q_table_cuda(ptr, next_q_value, state_x, state_y, action, next_state_x, next_state_y, reward, learning_rate, discount_factor);

    return result_array;
}

py::array_t<int> randomizeZeroswrapper(py::array_t<int> result_array, float percentage, unsigned long long seed) {
    auto buf1 = result_array.request();

    int x = result_array.shape()[0];
    int y = result_array.shape()[1];

    int* A = (int*)buf1.ptr;

    randomizeZerosCuda(A, x, y, percentage, seed);

    return result_array;
}

PYBIND11_MODULE(cu_matrix_add, m) {
    m.def("epsilon_greedy_cuda", &py_epsilonGreedyCUDA, "Compute epsilon-greedy exploration rates using CUDA");
    m.def("random_array", &randomArrayWrapper, "Generate a random array of 1's and 0's using CUDA");
    m.def("update_q_table_gpu", &update_q_table_gpu, "Update Q-table on GPU");
    m.def("randomizeZerosCuda", &randomizeZeroswrapper, "Randomly turn a percentage of 0's to 2's with CUDA");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

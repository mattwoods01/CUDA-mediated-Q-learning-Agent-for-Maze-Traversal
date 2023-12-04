/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void epsilonGreedyCUDA(float* exploration_rates, int num_episodes, float exploration_start, float exploration_end);
extern void randomArrayCuda(int* array, int height, int width, unsigned long long seed);
extern void qLearningCUDA(float* q_table, int state, int action, int next_state, float reward, float learning_rate, float discount_factor);


namespace py = pybind11;


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

py::array_t<float> qLearningWrapper(py::array_t<float, py::array::c_style | py::array::forcecast> q_table, int state, int action, int next_state, float reward, float learning_rate, float discount_factor) {
    // Get a pointer to the data in the NumPy array
    float* q_table_ptr = static_cast<float*>(q_table.request().ptr);

    // Call the Q-learning CUDA function
    qLearningCUDA(q_table_ptr, state, action, next_state, reward, learning_rate, discount_factor);

    return q_table;
}


PYBIND11_MODULE(cu_matrix_add, m) {
    m.def("epsilon_greedy_cuda", &py_epsilonGreedyCUDA, "Compute epsilon-greedy exploration rates using CUDA");
    m.def("random_array", &randomArrayWrapper, "Generate a random array of 1's and 0's using CUDA");
    m.def("q_learning_cuda", &qLearningWrapper, "Perform Q-learning update using CUDA");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

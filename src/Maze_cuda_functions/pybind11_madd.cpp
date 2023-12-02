/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_madd(int* A, int* B, int* C, int M, int N);

namespace py = pybind11;


py::array_t<int> madd_wrapper(py::array_t<int> a1, py::array_t<int> a2) 
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];
	printf("M=%d, N=%d\n", M, N);

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(int),     /* Size of one item */
		py::format_descriptor<int>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M},  /* Number of elements for each dimension */
		{ sizeof(int)*M, sizeof(int) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	int* A = (int*)buf1.ptr;
	int* B = (int*)buf2.ptr;
	int* C = (int*)buf3.ptr;

	cu_madd(A, B, C, M, N);

    return result;
}



PYBIND11_MODULE(cu_matrix_add, m) {
    m.def("madd", &madd_wrapper, "Add two NumPy arrays");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

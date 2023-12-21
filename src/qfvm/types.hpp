#pragma once

#include <Eigen/Core>
#include <complex>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _MSC_VER
using omp_i = signed long long;
#else
using omp_i = size_t;
#endif

#ifdef _MSC_VER
#else
#ifndef __AVX2__
#undef USE_SIMD
#endif
#endif

typedef unsigned int uint;
using pos_t = uint;
using data_t = double;
using std::complex;
using std::string;
using std::vector;
using RowMatrixXcd = Eigen::Matrix<complex<double>, Eigen::Dynamic,
                                   Eigen::Dynamic, Eigen::RowMajor>;

const complex<double> imag_I = complex<double>(0, 1.);
const double PI = 3.14159265358979323846;

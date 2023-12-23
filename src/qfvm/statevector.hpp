#pragma once
#include "types.hpp"
#include "util.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#ifdef USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

template <class real_t = double> class StateVector {
private:
  uint num_;
  // classical bit
  uint cbit_num_;
  vector<uint> creg_;
  size_t size_;
  std::unique_ptr<complex<real_t>[]> data_;
  // random engine
  std::mt19937_64 rng_;

public:
  // construct function
  StateVector();
  explicit StateVector(uint num);
  explicit StateVector(complex<real_t>* data, size_t data_size);
  // move assign
  //  StateVector& operator=(StateVector&& other){
  //      if(this != &other){
  //          data_ = std::move(other.data_);
  //          creg_ = std::move(other.creg_);
  //          num_ = other.num_;
  //          cbit_num_ = other.cbit_num_;
  //          size_ = other.size_;

  //     }
  //     return *this;
  // }

  // Named gate function
  void apply_x(pos_t pos);
  void apply_y(pos_t pos);
  void apply_z(pos_t pos);
  void apply_h(pos_t pos);
  void apply_s(pos_t pos);
  void apply_sdag(pos_t pos);
  void apply_t(pos_t pos);
  void apply_tdag(pos_t pos);
  void apply_p(pos_t pos, real_t phase);
  void apply_rx(pos_t pos, real_t theta);
  void apply_ry(pos_t pos, real_t theta);
  void apply_rz(pos_t pos, real_t theta);
  void apply_cnot(pos_t control, pos_t targe);
  void apply_cz(pos_t control, pos_t targe);
  void apply_cp(pos_t control, pos_t targe, real_t phase);
  void apply_crx(pos_t control, pos_t targe, real_t theta);
  void apply_cry(pos_t control, pos_t targe, real_t theta);
  void apply_ccx(pos_t control1, pos_t control2, pos_t targe);
  void apply_swap(pos_t q1, pos_t q2);

  // General implementation
  // One-target gate, ctrl_num equal 2 represent multi-controlled gate
  template <int ctrl_num>
  void apply_one_targe_gate_general(vector<pos_t> const& posv,
                                    complex<double>* mat);
  template <int ctrl_num>
  void apply_one_targe_gate_diag(vector<pos_t> const& posv,
                                 complex<double>* mat);
  template <int ctrl_num>
  void apply_one_targe_gate_real(vector<pos_t> const& posv,
                                 complex<double>* mat);
  template <int ctrl_num>
  void apply_one_targe_gate_x(vector<pos_t> const& posv);

  // Multiple-target gate
  void apply_multi_targe_gate_general(vector<pos_t> const& posv,
                                      uint control_num,
                                      RowMatrixXcd const& mat);

  // Measure and Reset
  std::pair<uint, double> sample_measure_probs(vector<pos_t> const& qbits);
  vector<double> probabilities() const;
  void apply_diagonal_matrix(vector<pos_t> const& qbits,
                             vector<std::complex<double>> const& mdiag);
  void update(vector<pos_t> const& qbits, const uint final_state,
              const uint meas_state, const double meas_prob);
  void apply_measure(vector<pos_t> const& qbits, const vector<pos_t>& cbits);
  void apply_reset(vector<pos_t> const& qbits);

  // cif check
  bool check_cif(const vector<pos_t>& cbits, const uint condition);

  complex<real_t> operator[](size_t j) const;
  void set_num(uint num);
  void set_creg(uint num) {
    if (num > 0) {
      cbit_num_ = num;
      creg_.resize(cbit_num_, 0);
    } else {
      throw std::logic_error("The number of cbit must be positive.");
    }
  }

  vector<uint> creg() { return creg_; }

  void set_rng() {
    std::random_device rd;
    rng_.seed(rd());
  }

  void print_state();
  std::tuple<std::complex<real_t>*, size_t> move_data_to_python() {
    auto data_ptr = data_.release();
    return std::make_tuple(std::move(data_ptr), size_);
  }

  complex<real_t>* data() { return data_.get(); }
  size_t size() { return size_; }
  uint num() { return num_; }
  uint cbit_num() { return cbit_num_; }
};

//////// constructors ///////

template <class real_t>
StateVector<real_t>::StateVector(uint num) : num_(num), size_(1ULL << num) {
  data_ = std::make_unique<complex<real_t>[]>(size_);
  data_[0] = complex<real_t>(1., 0);
};

template <class real_t> StateVector<real_t>::StateVector() : StateVector(0) {}

template <class real_t>
StateVector<real_t>::StateVector(complex<real_t>* data, size_t data_size)
    : data_(data), size_(data_size) {
  num_ = static_cast<int>(std::log2(size_));
}

//// useful functions /////
template <class real_t>
std::complex<real_t> StateVector<real_t>::operator[](size_t j) const {
  return data_[j];
}

template <class real_t> void StateVector<real_t>::set_num(uint num) {
  if (num_ > 0) {
    // Initialized from statevector,
    // should not resize
    return;
  }
  num_ = num;

  if (size_ != 1ULL << num) {
    data_.reset();
    size_ = 1ULL << num;
    data_ = std::make_unique<complex<real_t>[]>(size_);
    data_[0] = complex<real_t>(1, 0);
  }
}
template <class real_t>
bool StateVector<real_t>::check_cif(const vector<pos_t>& cbits,
                                    const uint condition) {
  uint out = 0;
  for (uint i = 0; i < cbits.size(); i++) {
    out *= 2;
    out += creg_[cbits[i]];
  }
  return out == condition;
}

template <class real_t> void StateVector<real_t>::print_state() {
  std::cout << "state_data: ";
  for (auto i = 0; i < size_; i++) {
    std::cout << data_[i] << " ";
  }
  std::cout << std::endl;
}

////// apply gate ////////

template <class real_t> void StateVector<real_t>::apply_x(pos_t pos) {
  const size_t offset = 1 << pos;
  const size_t rsize = size_ >> 1;
  if (pos == 0) { // single step
#ifdef USE_SIMD
#pragma omp parallel for
    for (omp_i j = 0; j < size_; j += 2) {
      double* ptr = (double*)(data_.get() + j);
      __m256d data = _mm256_loadu_pd(ptr);
      data = _mm256_permute4x64_pd(data, 78);
      _mm256_storeu_pd(ptr, data);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < size_; j += 2) {
      std::swap(data_[j], data_[j + 1]);
    }
#endif
  } else {
#ifdef USE_SIMD
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);
      double* ptr0 = (double*)(data_.get() + i);
      double* ptr1 = (double*)(data_.get() + i + offset);
      __m256d data0 = _mm256_loadu_pd(ptr0);
      __m256d data1 = _mm256_loadu_pd(ptr1);
      _mm256_storeu_pd(ptr1, data0);
      _mm256_storeu_pd(ptr0, data1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);
      size_t i1 = i + 1;
      std::swap(data_[i], data_[i + offset]);
      std::swap(data_[i1], data_[i1 + offset]);
    }
#endif
  }
}

template <class real_t> void StateVector<real_t>::apply_y(pos_t pos) {
  const size_t offset = 1 << pos;
  const size_t rsize = size_ >> 1;
  const complex<real_t> im = imag_I;
  if (pos == 0) { // single step
#ifdef USE_SIMD
    __m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
#pragma omp parallel for
    for (omp_i j = 0; j < size_; j += 2) {
      double* ptr = (double*)(data_.get() + j);
      __m256d data = _mm256_loadu_pd(ptr);
      data = _mm256_permute4x64_pd(data, 27);
      data = _mm256_mul_pd(data, minus_half);
      _mm256_storeu_pd(ptr, data);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < size_; j += 2) {
      complex<real_t> temp = data_[j];
      data_[j] = -im * data_[j + 1];
      data_[j + 1] = im * temp;
    }
#endif
  } else {
#ifdef USE_SIMD
    __m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
    __m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);

#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);

      double* ptr0 = (double*)(data_.get() + i);
      double* ptr1 = (double*)(data_.get() + i + offset);
      __m256d data0 = _mm256_loadu_pd(ptr0);
      __m256d data1 = _mm256_loadu_pd(ptr1);
      data0 = _mm256_permute_pd(data0, 5);
      data1 = _mm256_permute_pd(data1, 5);
      data0 = _mm256_mul_pd(data0, minus_even);
      data1 = _mm256_mul_pd(data1, minus_odd);
      _mm256_storeu_pd(ptr1, data0);
      _mm256_storeu_pd(ptr0, data1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);
      size_t i1 = i + 1;
      complex<real_t> temp = data_[i];
      data_[i] = -im * data_[i + offset];
      data_[i + offset] = im * temp;
      complex<real_t> temp1 = data_[i1];
      data_[i1] = -im * data_[i1 + offset];
      data_[i1 + offset] = im * temp1;
    }
#endif
  }
}

template <class real_t> void StateVector<real_t>::apply_z(pos_t pos) {
  const size_t offset = 1 << pos;
  const size_t rsize = size_ >> 1;
  if (pos == 0) { // single step
#pragma omp parallel for
    for (omp_i j = 1; j < size_; j += 2) {
      data_[j] *= -1;
    }
  } else {
#ifdef USE_SIMD
    __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);
      double* ptr1 = (double*)(data_.get() + i + offset);
      __m256d data1 = _mm256_loadu_pd(ptr1);
      data1 = _mm256_mul_pd(data1, minus_one);
      _mm256_storeu_pd(ptr1, data1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = (j & (offset - 1)) | (j >> pos << pos << 1);
      data_[i + offset] *= -1;
      data_[i + offset + 1] *= -1;
    }
#endif
  }
}

template <class real_t> void StateVector<real_t>::apply_h(pos_t pos) {
  const double sqrt2inv = 1. / std::sqrt(2.);
  complex<double> mat[4] = {sqrt2inv, sqrt2inv, sqrt2inv, -sqrt2inv};
  apply_one_targe_gate_real<0>(vector<pos_t>{pos}, mat);
}

template <class real_t> void StateVector<real_t>::apply_s(pos_t pos) {
  complex<double> mat[2] = {1., imag_I};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t> void StateVector<real_t>::apply_sdag(pos_t pos) {
  complex<double> mat[2] = {1., -imag_I};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t> void StateVector<real_t>::apply_t(pos_t pos) {
  complex<double> p = imag_I * PI / 4.;
  complex<double> mat[2] = {1., std::exp(p)};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t> void StateVector<real_t>::apply_tdag(pos_t pos) {
  complex<double> p = -imag_I * PI / 4.;
  complex<double> mat[2] = {1., std::exp(p)};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_p(pos_t pos, real_t phase) {
  complex<double> p = imag_I * phase;
  complex<double> mat[2] = {1., std::exp(p)};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_rx(pos_t pos, real_t theta) {
  complex<double> mat[4] = {std::cos(theta / 2), -imag_I * std::sin(theta / 2),
                            -imag_I * std::sin(theta / 2), std::cos(theta / 2)};
  apply_one_targe_gate_general<0>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_ry(pos_t pos, real_t theta) {
  complex<double> mat[4] = {std::cos(theta / 2), -std::sin(theta / 2),
                            std::sin(theta / 2), std::cos(theta / 2)};
  apply_one_targe_gate_real<0>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_rz(pos_t pos, real_t theta) {
  complex<double> z0 = -imag_I * theta / 2.;
  complex<double> z1 = imag_I * theta / 2.;
  complex<double> mat[2] = {std::exp(z0), std::exp(z1)};
  apply_one_targe_gate_diag<0>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cnot(pos_t control, pos_t targe) {
  apply_one_targe_gate_x<1>(vector<pos_t>{control, targe});
}

template <class real_t>
void StateVector<real_t>::apply_cz(pos_t control, pos_t targe) {
  complex<double> mat[2] = {1., -1.};
  apply_one_targe_gate_diag<1>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cp(pos_t control, pos_t targe, real_t phase) {
  complex<double> p = imag_I * phase;
  complex<double> mat[2] = {1., std::exp(p)};
  apply_one_targe_gate_diag<1>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_crx(pos_t control, pos_t targe, real_t theta) {
  complex<double> mat[4] = {std::cos(theta / 2), -imag_I * std::sin(theta / 2),
                            -imag_I * std::sin(theta / 2), std::cos(theta / 2)};

  apply_one_targe_gate_general<1>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cry(pos_t control, pos_t targe, real_t theta) {
  complex<double> mat[4] = {std::cos(theta / 2), -std::sin(theta / 2),
                            std::sin(theta / 2), std::cos(theta / 2)};

  apply_one_targe_gate_real<1>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_ccx(pos_t control1, pos_t control2,
                                    pos_t targe) {
  apply_one_targe_gate_x<2>(vector<pos_t>{control1, control2, targe});
}

/////// General implementation /////////

template <class real_t>
template <int ctrl_num>
void StateVector<real_t>::apply_one_targe_gate_general(
    vector<pos_t> const& posv, complex<double>* mat) {
  std::function<size_t(size_t)> getind_func_near;
  std::function<size_t(size_t)> getind_func;
  size_t rsize;
  size_t offset;
  size_t targe;
  size_t control = 0;
  size_t setbit;
  size_t poffset;
  bool has_control = false;
  vector<pos_t> posv_sorted = posv;
  if (ctrl_num == 0) {
    targe = posv[0];
    offset = 1ll << targe;
    rsize = size_ >> 1;
    getind_func_near = [&](size_t j) -> size_t { return 2 * j; };

    getind_func = [&](size_t j) -> size_t {
      return (j & (offset - 1)) | (j >> targe << targe << 1);
    };

  } else if (ctrl_num == 1) {
    has_control = true;
    control = posv[0];
    targe = posv[1];
    offset = 1ll << targe;
    setbit = 1ll << control;
    if (control > targe) {
      control--;
    }
    poffset = 1ll << control;
    rsize = size_ >> 2;
    getind_func = [&](size_t j) -> size_t {
      size_t i = (j >> control << (control + 1)) | (j & (poffset - 1));
      i = (i >> targe << (targe + 1)) | (i & (offset - 1)) | setbit;
      return i;
    };

    getind_func_near = getind_func;

  } else if (ctrl_num == 2) {
    has_control = true;
    control = *min_element(posv.begin(), posv.end() - 1);
    targe = *(posv.end() - 1);
    offset = 1ll << targe;
    sort(posv_sorted.begin(), posv_sorted.end());
    rsize = size_ >> posv.size();
    getind_func = [&](size_t j) -> size_t {
      size_t i = j;
      for (size_t k = 0; k < posv.size(); k++) {
        size_t _pos = posv_sorted[k];
        i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
      }
      for (size_t k = 0; k < posv.size() - 1; k++) {
        i |= 1ll << posv[k];
      }
      return i;
    };
    getind_func_near = getind_func;
  }

  const complex<real_t> mat00 = mat[0];
  const complex<real_t> mat01 = mat[1];
  const complex<real_t> mat10 = mat[2];
  const complex<real_t> mat11 = mat[3];
  if (targe == 0) {
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func_near(j);
      complex<real_t> temp = data_[i];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + 1];
      data_[i + 1] = mat10 * temp + mat11 * data_[i + 1];
    }
  } else if (has_control && control == 0) { // single step
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func(j);
      complex<real_t> temp = data_[i];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + offset];
      data_[i + offset] = mat10 * temp + mat11 * data_[i + offset];
    }

  } else { // unroll to 2
#ifdef USE_SIMD
    __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(), mat[0].real(),
                                   mat[0].real());
    __m256d m_00im = _mm256_set_pd(mat[0].imag(), -mat[0].imag(), mat[0].imag(),
                                   -mat[0].imag());
    __m256d m_01re = _mm256_set_pd(mat[1].real(), mat[1].real(), mat[1].real(),
                                   mat[1].real());
    __m256d m_01im = _mm256_set_pd(mat[1].imag(), -mat[1].imag(), mat[1].imag(),
                                   -mat[1].imag());

    __m256d m_10re = _mm256_set_pd(mat[2].real(), mat[2].real(), mat[2].real(),
                                   mat[2].real());
    __m256d m_10im = _mm256_set_pd(mat[2].imag(), -mat[2].imag(), mat[2].imag(),
                                   -mat[2].imag());
    __m256d m_11re = _mm256_set_pd(mat[3].real(), mat[3].real(), mat[3].real(),
                                   mat[3].real());
    __m256d m_11im = _mm256_set_pd(mat[3].imag(), -mat[3].imag(), mat[3].imag(),
                                   -mat[3].imag());
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);

      double* p0 = (double*)(data_.get() + i);
      double* p1 = (double*)(data_.get() + i + offset);
      // load data
      __m256d data0 = _mm256_loadu_pd(p0); // lre_0, lim_0, rre_0, rim_0
      __m256d data1 = _mm256_loadu_pd(p1); // lre_1, lim_1, rre_1, rim_1
      __m256d data0_p = _mm256_permute_pd(data0, 5);
      __m256d data1_p = _mm256_permute_pd(data1, 5);

      // row0
      __m256d temp00re = _mm256_mul_pd(m_00re, data0);
      __m256d temp00im = _mm256_mul_pd(m_00im, data0_p);
      __m256d temp00 = _mm256_add_pd(temp00re, temp00im);
      __m256d temp01re = _mm256_mul_pd(m_01re, data1);
      __m256d temp01im = _mm256_mul_pd(m_01im, data1_p);
      __m256d temp01 = _mm256_add_pd(temp01re, temp01im);
      __m256d temp0 = _mm256_add_pd(temp00, temp01);

      // row1
      __m256d temp10re = _mm256_mul_pd(m_10re, data0);
      __m256d temp10im = _mm256_mul_pd(m_10im, data0_p);
      __m256d temp10 = _mm256_add_pd(temp10re, temp10im);
      __m256d temp11re = _mm256_mul_pd(m_11re, data1);
      __m256d temp11im = _mm256_mul_pd(m_11im, data1_p);
      __m256d temp11 = _mm256_add_pd(temp11re, temp11im);
      __m256d temp1 = _mm256_add_pd(temp10, temp11);

      _mm256_storeu_pd(p0, temp0);
      _mm256_storeu_pd(p1, temp1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);
      size_t i1 = i + 1;
      complex<real_t> temp = data_[i];
      complex<real_t> temp1 = data_[i1];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + offset];
      data_[i + offset] = mat10 * temp + mat11 * data_[i + offset];
      data_[i1] = mat00 * data_[i1] + mat01 * data_[i1 + offset];
      data_[i1 + offset] = mat10 * temp1 + mat11 * data_[i1 + offset];
    }
#endif
  }
}

template <class real_t>
template <int ctrl_num>
void StateVector<real_t>::apply_one_targe_gate_x(vector<pos_t> const& posv) {
  std::function<size_t(size_t)> getind_func_near;
  std::function<size_t(size_t)> getind_func;
  size_t rsize;
  size_t offset;
  size_t targe;
  size_t control;
  size_t setbit;
  size_t poffset;
  vector<pos_t> posv_sorted = posv;
  bool has_control = false;
  if (ctrl_num == 0) {
    targe = posv[0];
    offset = 1ll << targe;
    rsize = size_ >> 1;
    getind_func_near = [&](size_t j) -> size_t { return 2 * j; };

    getind_func = [&](size_t j) -> size_t {
      return (j & (offset - 1)) | (j >> targe << targe << 1);
    };

  } else if (ctrl_num == 1) {
    has_control = true;
    control = posv[0];
    targe = posv[1];
    offset = 1ll << targe;
    setbit = 1ll << control;
    if (control > targe) {
      control--;
    }
    poffset = 1ll << control;
    rsize = size_ >> 2;
    getind_func = [&](size_t j) -> size_t {
      size_t i = (j >> control << (control + 1)) | (j & (poffset - 1));
      i = (i >> targe << (targe + 1)) | (i & (offset - 1)) | setbit;
      return i;
    };
    getind_func_near = getind_func;
  } else if (ctrl_num == 2) {
    has_control = true;
    control = *min_element(posv.begin(), posv.end() - 1);
    targe = *(posv.end() - 1);
    offset = 1ll << targe;
    sort(posv_sorted.begin(), posv_sorted.end());
    rsize = size_ >> posv.size();

    getind_func = [&](size_t j) -> size_t {
      size_t i = j;
      for (size_t k = 0; k < posv.size(); k++) {
        size_t _pos = posv_sorted[k];
        i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
      }
      for (size_t k = 0; k < posv.size() - 1; k++) {
        i |= 1ll << posv[k];
      }
      return i;
    };
    getind_func_near = getind_func;
  }

  if (targe == 0) {
#ifdef USE_SIMD
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func_near(j);
      double* ptr = (double*)(data_.get() + i);
      __m256d data = _mm256_loadu_pd(ptr);
      data = _mm256_permute4x64_pd(data, 78);
      _mm256_storeu_pd(ptr, data);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func(j);
      std::swap(data_[i], data_[i + 1]);
    }
#endif
  } else if (has_control && control == 0) { // single step
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func(j);
      std::swap(data_[i], data_[i + offset]);
    }

  } else { // unroll to 2
#ifdef USE_SIMD
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);
      double* ptr0 = (double*)(data_.get() + i);
      double* ptr1 = (double*)(data_.get() + i + offset);
      __m256d data0 = _mm256_loadu_pd(ptr0);
      __m256d data1 = _mm256_loadu_pd(ptr1);
      _mm256_storeu_pd(ptr1, data0);
      _mm256_storeu_pd(ptr0, data1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);
      size_t i1 = i + 1;
      std::swap(data_[i], data_[i + offset]);
      std::swap(data_[i1], data_[i1 + offset]);
    }
#endif
  }
}

template <class real_t>
template <int ctrl_num>
void StateVector<real_t>::apply_one_targe_gate_real(vector<pos_t> const& posv,
                                                    complex<double>* mat) {
  std::function<size_t(size_t)> getind_func_near;
  std::function<size_t(size_t)> getind_func;
  size_t rsize;
  size_t offset;
  size_t targe;
  size_t control = 0;
  size_t setbit;
  size_t poffset;
  bool has_control = false;
  vector<pos_t> posv_sorted = posv;
  if (ctrl_num == 0) {
    targe = posv[0];
    offset = 1ll << targe;
    rsize = size_ >> 1;
    getind_func_near = [&](size_t j) -> size_t { return 2 * j; };

    getind_func = [&](size_t j) -> size_t {
      return (j & (offset - 1)) | (j >> targe << targe << 1);
    };

  } else if (ctrl_num == 1) {
    has_control = true;
    control = posv[0];
    targe = posv[1];
    offset = 1ll << targe;
    setbit = 1ll << control;
    if (control > targe) {
      control--;
    }
    poffset = 1ll << control;
    rsize = size_ >> 2;
    getind_func = [&](size_t j) -> size_t {
      size_t i = (j >> control << (control + 1)) | (j & (poffset - 1));
      i = (i >> targe << (targe + 1)) | (i & (offset - 1)) | setbit;
      return i;
    };

    getind_func_near = getind_func;
  } else if (ctrl_num == 2) {
    has_control = true;
    control = *min_element(posv.begin(), posv.end() - 1);
    targe = *(posv.end() - 1);
    offset = 1ll << targe;
    sort(posv_sorted.begin(), posv_sorted.end());
    rsize = size_ >> posv.size();
    getind_func = [&](size_t j) -> size_t {
      size_t i = j;
      for (size_t k = 0; k < posv.size(); k++) {
        size_t _pos = posv_sorted[k];
        i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
      }
      for (size_t k = 0; k < posv.size() - 1; k++) {
        i |= 1ll << posv[k];
      }
      return i;
    };
    getind_func_near = getind_func;
  }

  const double mat00 = mat[0].real();
  const double mat01 = mat[1].real();
  const double mat10 = mat[2].real();
  const double mat11 = mat[3].real();
  if (targe == 0) {
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func_near(j);
      complex<real_t> temp = data_[i];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + 1];
      data_[i + 1] = mat10 * temp + mat11 * data_[i + 1];
    }

  } else if (has_control && control == 0) { // single step

#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func(j);
      complex<real_t> temp = data_[i];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + offset];
      data_[i + offset] = mat10 * temp + mat11 * data_[i + offset];
    }
  } else { // unroll to 2
#ifdef USE_SIMD
    __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(), mat[0].real(),
                                   mat[0].real());
    __m256d m_01re = _mm256_set_pd(mat[1].real(), mat[1].real(), mat[1].real(),
                                   mat[1].real());
    __m256d m_10re = _mm256_set_pd(mat[2].real(), mat[2].real(), mat[2].real(),
                                   mat[2].real());
    __m256d m_11re = _mm256_set_pd(mat[3].real(), mat[3].real(), mat[3].real(),
                                   mat[3].real());
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);

      double* p0 = (double*)(data_.get() + i);
      double* p1 = (double*)(data_.get() + i + offset);
      // load data
      __m256d data0 = _mm256_loadu_pd(p0); // lre_0, lim_0, rre_0, rim_0
      __m256d data1 = _mm256_loadu_pd(p1); // lre_1, lim_1, rre_1, rim_1
      __m256d data0_p = _mm256_permute_pd(data0, 5);
      __m256d data1_p = _mm256_permute_pd(data1, 5);

      // row0
      __m256d temp00re = _mm256_mul_pd(m_00re, data0);
      __m256d temp01re = _mm256_mul_pd(m_01re, data1);
      __m256d temp0 = _mm256_add_pd(temp00re, temp01re);

      // row1
      __m256d temp10re = _mm256_mul_pd(m_10re, data0);
      __m256d temp11re = _mm256_mul_pd(m_11re, data1);
      __m256d temp1 = _mm256_add_pd(temp10re, temp11re);

      _mm256_storeu_pd(p0, temp0);
      _mm256_storeu_pd(p1, temp1);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);
      size_t i1 = i + 1;
      complex<real_t> temp = data_[i];
      complex<real_t> temp1 = data_[i1];
      data_[i] = mat00 * data_[i] + mat01 * data_[i + offset];
      data_[i + offset] = mat10 * temp + mat11 * data_[i + offset];
      data_[i1] = mat00 * data_[i1] + mat01 * data_[i1 + offset];
      data_[i1 + offset] = mat10 * temp1 + mat11 * data_[i1 + offset];
    }
#endif
  }
}

template <class real_t>
template <int ctrl_num>
void StateVector<real_t>::apply_one_targe_gate_diag(vector<pos_t> const& posv,
                                                    complex<double>* mat) {
  std::function<size_t(size_t)> getind_func_near;
  std::function<size_t(size_t)> getind_func;
  size_t rsize;
  size_t offset;
  size_t targe;
  size_t control = 0;
  size_t setbit;
  size_t poffset;
  bool has_control = false;
  vector<pos_t> posv_sorted = posv;
  if (ctrl_num == 0) {
    targe = posv[0];
    offset = 1ll << targe;
    rsize = size_ >> 1;
    getind_func_near = [&](size_t j) -> size_t { return 2 * j; };

    getind_func = [&](size_t j) -> size_t {
      return (j & (offset - 1)) | (j >> targe << targe << 1);
    };

  } else if (ctrl_num == 1) {

    has_control = true;
    control = posv[0];
    targe = posv[1];
    offset = 1ll << targe;
    setbit = 1ll << control;
    if (control > targe) {
      control--;
    }
    poffset = 1ll << control;
    rsize = size_ >> 2;
    getind_func = [&](size_t j) -> size_t {
      size_t i = (j >> control << (control + 1)) | (j & (poffset - 1));
      i = (i >> targe << (targe + 1)) | (i & (offset - 1)) | setbit;
      return i;
    };

    getind_func_near = getind_func;

  } else if (ctrl_num == 2) {
    has_control = true;
    control = *min_element(posv.begin(), posv.end() - 1);
    targe = *(posv.end() - 1);
    offset = 1ll << targe;
    sort(posv_sorted.begin(), posv_sorted.end());
    rsize = size_ >> posv.size();
    getind_func = [&](size_t j) -> size_t {
      size_t i = j;
      for (size_t k = 0; k < posv.size(); k++) {
        size_t _pos = posv_sorted[k];
        i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
      }
      for (size_t k = 0; k < posv.size() - 1; k++) {
        i |= 1ll << posv[k];
      }
      return i;
    };
    getind_func_near = getind_func;
  }

  if (targe == 0) {
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func_near(j);
      data_[i] *= mat[0];
      data_[i + 1] *= mat[1];
    }

  } else if (has_control && control == 0) { // single step

#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j++) {
      size_t i = getind_func(j);
      complex<real_t> temp = data_[i];
      data_[i] *= mat[0];
      data_[i + offset] *= mat[1];
    }

  } else { // unroll to 2
#ifdef USE_SIMD
    __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(), mat[0].real(),
                                   mat[0].real());
    __m256d m_00im = _mm256_set_pd(mat[0].imag(), -mat[0].imag(), mat[0].imag(),
                                   -mat[0].imag());
    __m256d m_11re = _mm256_set_pd(mat[1].real(), mat[1].real(), mat[1].real(),
                                   mat[1].real());
    __m256d m_11im = _mm256_set_pd(mat[1].imag(), -mat[1].imag(), mat[1].imag(),
                                   -mat[1].imag());
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);

      double* p0 = (double*)(data_.get() + i);
      double* p1 = (double*)(data_.get() + i + offset);

      // load data
      __m256d data0 = _mm256_loadu_pd(p0); // lre_0, lim_0, rre_0, rim_0
      __m256d data1 = _mm256_loadu_pd(p1); // lre_1, lim_1, rre_1, rim_1
      __m256d data0_p = _mm256_permute_pd(data0, 5);
      __m256d data1_p = _mm256_permute_pd(data1, 5);

      // row0
      __m256d temp00re = _mm256_mul_pd(m_00re, data0);
      __m256d temp00im = _mm256_mul_pd(m_00im, data0_p);
      __m256d temp00 = _mm256_add_pd(temp00re, temp00im);

      // row1
      __m256d temp11re = _mm256_mul_pd(m_11re, data1);
      __m256d temp11im = _mm256_mul_pd(m_11im, data1_p);
      __m256d temp11 = _mm256_add_pd(temp11re, temp11im);

      _mm256_storeu_pd(p0, temp00);
      _mm256_storeu_pd(p1, temp11);
    }
#else
#pragma omp parallel for
    for (omp_i j = 0; j < rsize; j += 2) {
      size_t i = getind_func(j);
      size_t i1 = i + 1;
      data_[i] *= mat[0];
      data_[i + offset] *= mat[1];
      data_[i1] *= mat[0];
      data_[i1 + offset] *= mat[1];
    }
#endif
  }
}

template <class real_t>
void StateVector<real_t>::apply_multi_targe_gate_general(
    vector<pos_t> const& posv, uint control_num, RowMatrixXcd const& mat) {
  auto posv_sorted = posv;
  auto targs = vector<pos_t>(posv.begin() + control_num, posv.end());
  sort(posv_sorted.begin(), posv_sorted.end());
  size_t rsize = size_ >> posv.size();
  uint targe_num = targs.size();
  size_t matsize = 1 << targe_num;
  std::vector<uint> targ_mask(matsize);
  // create target mask
  for (size_t m = 0; m < matsize; m++) {
    for (size_t j = 0; j < targe_num; j++) {
      if ((m >> j) & 1) {
        auto mask_pos = targs[j];
        targ_mask[m] |= 1ll << mask_pos;
      }
    }
  }

  // apply matrix
// TODO: Disalbe Parallel when matsize is very large
#pragma omp parallel for
  for (omp_i j = 0; j < rsize; j++) {
    size_t i = j;
    // Insert zeros
    for (size_t k = 0; k < posv.size(); k++) {
      size_t _pos = posv_sorted[k];
      i = (i & ((1ll << _pos) - 1)) | (i >> _pos << _pos << 1);
    }
    // Set control
    for (size_t k = 0; k < control_num; k++) {
      i |= 1ll << posv[k];
    }

    // load block vector
    Eigen::VectorXcd vec_block(matsize);
    for (size_t m = 0; m < matsize; m++) {
      vec_block(m) = data_[i | targ_mask[m]];
      auto ele = vec_block(m);
    }

    // Eigen matrix multiply
    vec_block = mat * vec_block;

    // write back
    for (size_t m = 0; m < matsize; m++) {
      data_[i | targ_mask[m]] = vec_block(m);
    }
  }
}

uint index0(vector<pos_t> const& qubits_sorted, const uint k) {
  uint lowbits, retval = k;
  for (size_t j = 0; j < qubits_sorted.size(); j++) {
    lowbits = retval & ((1LL << qubits_sorted[j]) - 1);
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

using indexes_t = vector<uint>;
inline indexes_t indexes(vector<pos_t> const& qbits,
                         vector<pos_t> const& qubits_sorted, const uint k) {
  const auto N = qubits_sorted.size();
  indexes_t ret(1LL << N, 0);
  // Get index0
  ret[0] = index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = 1LL << i;
    const auto bit = 1ll << qbits[i];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

template <class real_t>
vector<double> StateVector<real_t>::probabilities() const {
  const int len = 1LL << num_;
  vector<double> probs(len, 0.);
#pragma omp parallel for
  for (int j = 0; j < len; j++) {
    probs[j] = std::real(data_[j] * std::conj(data_[j]));
  }
  return probs;
}

vector<std::complex<double>> convert(const vector<std::complex<double>>& v) {
  vector<std::complex<double>> ret(v.size(), 0.);
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
}

template <class real_t>
void StateVector<real_t>::apply_diagonal_matrix(
    vector<pos_t> const& qbits, vector<std::complex<double>> const& diag) {
  // just one qubit
  if (qbits.size() == 1) {
    const uint qubit = qbits[0];
    vector<pos_t> qbit0{qubit};
    if (diag[0] == 1.0) { // [[1, 0], [0, z]] matrix
      if (diag[1] == 1.0)
        return;                                       // Identity
      if (diag[1] == std::complex<double>(0., -1.)) { // [[1, 0], [0, -i]]
        auto func = [&](const indexes_t& inds) -> void {
          const auto k = inds[1];
          double cache = data_[k].imag();
          data_[k].imag(data_[k].real() * -1.);
          data_[k].real(cache);
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      if (diag[1] == std::complex<double>(0., 1.)) {
        // [[1, 0], [0, i]]
        auto func = [&](const indexes_t& inds) -> void {
          const auto k = inds[1];
          double cache = data_[k].imag();
          data_[k].imag(data_[k].real());
          data_[k].real(cache * -1.);
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      if (diag[0] == 0.0) {
        // [[1, 0], [0, 0]]
        auto func = [&](const indexes_t& inds) -> void {
          data_[inds[1]] = 0.0;
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      // general [[1, 0], [0, z]]
      auto func = [&](const indexes_t& inds,
                      const vector<std::complex<double>>& _mat) -> void {
        const auto k = inds[1];
        data_[k] *= _mat[1];
      };
#pragma omp parallel for
      for (int k = 0; k < (size_ >> 1); k += 1) {
        const auto inds = indexes(qbit0, qbit0, k);
        func(inds, convert(diag));
      }
      return;
    } else if (diag[1] == 1.0) {
      // [[z, 0], [0, 1]] matrix
      if (diag[0] == std::complex<double>(0., -1.)) {
        // [[-i, 0], [0, 1]]
        auto func = [&](const indexes_t& inds) -> void {
          const auto k = inds[1];
          double cache = data_[k].imag();
          data_[k].imag(data_[k].real() * -1.);
          data_[k].real(cache);
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      if (diag[0] == std::complex<double>(0., 1.)) {
        // [[i, 0], [0, 1]]
        auto func = [&](const indexes_t& inds) -> void {
          const auto k = inds[1];
          double cache = data_[k].imag();
          data_[k].imag(data_[k].real());
          data_[k].real(cache * -1.);
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      if (diag[0] == 0.0) {
        // [[0, 0], [0, 1]]
        auto func = [&](const indexes_t& inds) -> void {
          data_[inds[0]] = 0.0;
        };
#pragma omp parallel for
        for (int k = 0; k < (size_ >> 1); k += 1) {
          const auto inds = indexes(qbit0, qbit0, k);
          func(inds);
        }
        return;
      }
      // general [[z, 0], [0, 1]]
      auto func = [&](const indexes_t& inds,
                      const vector<std::complex<double>>& _mat) -> void {
        const auto k = inds[0];
        data_[k] *= _mat[0];
      };
#pragma omp parallel for
      for (int k = 0; k < (size_ >> 1); k += 1) {
        const auto inds = indexes(qbit0, qbit0, k);
        func(inds, convert(diag));
      }
      return;
    } else {
      // Lambda function for diagonal matrix multiplication
      auto func = [&](const indexes_t& inds,
                      const vector<std::complex<double>>& _mat) -> void {
        const auto k0 = inds[0];
        const auto k1 = inds[1];
        data_[k0] *= _mat[0];
        data_[k1] *= _mat[1];
      };
#pragma omp parallel for
      for (int k = 0; k < (size_ >> 1); k += 1) {
        const auto inds = indexes(qbit0, qbit0, k);
        func(inds, convert(diag));
      }
    }
    return;
  }
  const uint N = qbits.size();
  auto func = [&](const indexes_t& inds,
                  const vector<std::complex<double>>& _diag) -> void {
    for (int i = 0; i < 2; ++i) {
      const int k = inds[i];
      int iv = 0;
      for (int j = 0; j < N; j++) {
        if ((k & (1ULL << qbits[j])) != 0)
          iv += (1ULL << j);
      }
      if (_diag[iv] != (double)1.0)
        data_[k] *= _diag[iv];
    }
  };
  // apply func
  vector<pos_t> qbit0{qbits[0]};
#pragma omp parallel for
  for (int k = 0; k < (size_ >> 1); k += 1) {
    const auto inds = indexes(qbit0, qbit0, k);
    func(inds, convert(diag));
  }
}

template <class real_t>
void StateVector<real_t>::update(vector<pos_t> const& qbits,
                                 const uint final_state, const uint meas_state,
                                 const double meas_prob) {
  const uint dim = 1ULL << qbits.size();
  vector<std::complex<double>> matdiag(dim, 0.);
  matdiag[meas_state] = 1. / std::sqrt(meas_prob);
  apply_diagonal_matrix(qbits, matdiag);

  // TODO: Add reset
  //  for reset
  if (final_state != meas_state) {
    if (qbits.size() == 1) {
      // apply a x gate
      apply_x(qbits[0]);
    } else {
      // Diagonal matrix for projecting and renormalizing to measurement outcome
      vector<std::complex<double>> perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (uint j = 0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      const uint N = qbits.size();
      const uint DIM = 1ULL << N;
      auto func = [&](const indexes_t& inds,
                      const vector<std::complex<double>>& _mat) -> void {
        // std::array<std::complex<double>, 1ULL << N > cache;
        vector<std::complex<double>> cache(1ULL << N, 0.);
        for (uint i = 0; i < DIM; i++) {
          const auto ii = inds[i];
          cache[i] = data_[ii];
          data_[ii] = 0.;
        }
        for (uint i = 0; i < DIM; i++)
          for (uint j = 0; j < DIM; j++)
            data_[inds[i]] += _mat[i + DIM * j] * cache[j];
      };
      vector<pos_t> qs(qbits.begin(), qbits.end());
      vector<pos_t> qs_sorted(qs.begin(), qs.end());
      std::sort(qs_sorted.begin(), qs_sorted.end());
      uint END = size_ >> qs.size();
#pragma omp parallel for
      for (int k = 0; k < END; k += 1) {
        const auto inds = indexes(qs, qs_sorted, k);
        func(inds, convert(perm));
      }
    }
  }
}

template <typename T> void printVector(const std::vector<T>& vec) {
  for (const T& element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

template <class real_t>
std::pair<uint, double>
StateVector<real_t>::sample_measure_probs(vector<pos_t> const& qbits) {
  // 1. caculate actual measurement outcome
  const int64_t N = qbits.size();
  const int64_t DIM = 1LL << N;
  const int64_t END = 1LL << (num_ - N);
  vector<double> probs(DIM, 0.);
  vector<uint> qubits_sorted(qbits.begin(), qbits.end());

  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((num_ == N) && (qubits_sorted == qbits)) {
    probs = probabilities();
  } else {
    vector<double> probs_private(DIM, 0.);
#pragma omp parallel for
    for (int64_t k = 0; k < END; k++) {
      auto idx = indexes(qbits, qubits_sorted, k);
      // std::cout<<"indexes"<<k<<": ";
      // printVector(idx);
      for (int64_t m = 0; m < DIM; ++m) {
        double local_prob = std::real(data_[idx[m]] * std::conj(data_[idx[m]]));
#pragma omp critical
        probs_private[m] += local_prob;
      }
    }
// std::cout<<"probs_private:";
// printVector(probs_private);
#pragma omp critical
    for (int64_t m = 0; m < DIM; ++m) {
      probs[m] += probs_private[m];
    }
  }
  set_rng();
  // std::cout<<"probs:";
  // printVector(probs);
  uint outcome =
      std::discrete_distribution<uint>(probs.begin(), probs.end())(rng_);
  return std::make_pair(outcome, probs[outcome]);
}

// change to bit endian
vector<uint> int2vec(uint n, uint base) {
  vector<uint> ret;
  while (n >= base) {
    ret.push_back(n % base);
    n /= base;
  }
  ret.push_back(n);
  return ret;
}

template <class real_t>
void StateVector<real_t>::apply_measure(vector<pos_t> const& qbits,
                                        vector<pos_t> const& cbits) {
  // 1. caculate actual measurement outcome
  const auto meas = sample_measure_probs(qbits);
  // 2. update statevector
  update(qbits, meas.first, meas.first, meas.second);
  // 3. store measure
  vector<uint> outcome = int2vec(meas.first, 2);
  if (outcome.size() < qbits.size()) {
    outcome.resize(qbits.size());
  }
  for (uint j = 0; j < outcome.size(); j++) {
    creg_[cbits[j]] = outcome[j];
  }
}

template <class real_t>
void StateVector<real_t>::apply_reset(vector<pos_t> const& qbits) {
  const auto meas = sample_measure_probs(qbits);
  update(qbits, 0, meas.first, meas.second);
}

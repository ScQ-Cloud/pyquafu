//TODO: add swap, 2qgate(rzz, rxx, xy...), mcgate, cmtarge, mcmtarge.
#pragma once

#include "types.hpp"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <functional>
#include <algorithm>
#ifdef USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

template <class real_t = double>
class StateVector{
    private:
        uint num_;
        size_t size_;
        vector<complex<real_t>> data_;

    public:
        //construct function
        StateVector();
        explicit StateVector(uint num);

        //Gate function
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
        void apply_crx(pos_t control, pos_t targe,  real_t theta);
        void apply_cry(pos_t control, pos_t targe,  real_t theta);
        void apply_ccx(pos_t control1, pos_t control2, pos_t targe);
        
        template<int num>
        void apply_one_targe_gate_general(vector<pos_t> const& posv, complex<double> *mat);
        template<int num>
        void apply_one_targe_gate_diag(vector<pos_t> const& posv, complex<double> *mat);
        template<int num>
        void apply_one_targe_gate_real(vector<pos_t> const& posv, complex<double> *mat);
        template<int num>
        void apply_one_targe_gate_x(vector<pos_t> const& posv);


        void apply_mctrl_1b_gate(pos_t *ctrl_list, pos_t targe, complex<double> *mat);
        void apply_ctrl_mb_gate(pos_t control, pos_t *targe_list, complex<double> *mat);
      

        complex<real_t> operator[] (size_t j) const ;
        void set_num(uint num);
        vector<complex<real_t>> move_data(){ return std::move(data_); }
        void print_state(); 
};


//////// constructors ///////

template <class real_t>
StateVector<real_t>::StateVector(uint num) 
: num_(num), 
size_(std::pow(2, num)),
data_(size_)
{ 
    data_[0] = complex<real_t>(1., 0);
};

template <class real_t>
StateVector<real_t>::StateVector() : StateVector(0){ }


//// useful functions /////
template <class real_t>
std::complex<real_t> StateVector<real_t>::operator[] (size_t j) const{
    return data_[j];
}

template <class real_t>
void StateVector<real_t>::set_num(uint num){
            num_ = num;
            size_ = std::pow(2, num_);
            data_.resize(size_);
        }  

template <class real_t>
void StateVector<real_t>::print_state(){
    for (auto i : data_){
        std::cout << i << std::endl;
    }
}


////// apply gate ////////

template <class real_t>
void StateVector<real_t>::apply_x(pos_t pos){
    const size_t offset = 1<<pos;
    const size_t rsize = size_>>1;
     if (pos == 0){ //single step
#ifdef USE_SIMD 
#pragma omp parallel for
         for(omp_i j = 0;j < size_;j+=2){
             double* ptr = (double*)(data_.data() + j);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data, 78); 
            _mm256_storeu_pd(ptr, data);
         }
#else
#pragma omp parallel for
            for(omp_i j = 0;j < size_;j+=2){
                std::swap(data_[j], data_[j+1]);
            }
#endif
     }
     else{
#ifdef USE_SIMD
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);
            double* ptr0 = (double*)(data_.data() + i);
            double* ptr1 = (double*)(data_.data() + i + offset);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);
            size_t i1 = i+1;
            std::swap(data_[i], data_[i+offset]);
            std::swap(data_[i1], data_[i1+offset]);
        }
#endif
     }
}

template <class real_t>
void StateVector<real_t>::apply_y(pos_t pos){
    const size_t offset = 1<<pos;
    const size_t rsize = size_>>1;
    const complex<real_t> im = imag_I;
     if (pos == 0){ //single step
#ifdef USE_SIMD
        __m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
#pragma omp parallel for
         for(omp_i j = 0;j < size_;j+=2){
             double* ptr = (double*)(data_.data() + j);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data, 27);
            data = _mm256_mul_pd(data, minus_half);
            _mm256_storeu_pd(ptr, data);
         }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < size_;j+=2){
            complex<real_t> temp = data_[j]; 
            data_[j] = -im*data_[j+1];
            data_[j+1] = im*temp;
        }
#endif
     }
     else{
#ifdef USE_SIMD
        __m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
        __m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
        
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);

            double* ptr0 = (double*)(data_.data() + i);
            double* ptr1 = (double*)(data_.data() + i + offset);
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
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);
            size_t i1 = i+1;
            complex<real_t> temp = data_[i]; 
            data_[i] = -im*data_[i+offset];
            data_[i+offset] = im*temp;
            complex<real_t> temp1 = data_[i1]; 
            data_[i1] = -im*data_[i1+offset];
            data_[i1+offset] = im*temp1;
        }
#endif
     }
}

template <class real_t>
void StateVector<real_t>::apply_z(pos_t pos){
    const size_t offset = 1<<pos;
    const size_t rsize = size_>>1;
    if (pos == 0){ //single step
#pragma omp parallel for
        for(omp_i j = 1;j < size_;j+=2){
            data_[j] *= -1;
        }
     }
     else{
#ifdef USE_SIMD
        __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);
            double* ptr1 = (double*)(data_.data() + i + offset);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            data1 = _mm256_mul_pd(data1, minus_one);
            _mm256_storeu_pd(ptr1, data1);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = (j&(offset-1)) | (j>>pos<<pos<<1);
            data_[i+offset] *= -1;
            data_[i+offset+1] *= -1;
        }
#endif
     }
}

template <class real_t>
void StateVector<real_t>::apply_h(pos_t pos){
    const double sqrt2inv = 1. / std::sqrt(2.);
    complex<double> mat[4] = {sqrt2inv, sqrt2inv, sqrt2inv, -sqrt2inv};
    apply_one_targe_gate_real<1>(vector<pos_t>{pos}, mat);

}

template <class real_t>
void StateVector<real_t>::apply_s(pos_t pos){
    complex<double> mat[2] = {1., imag_I};
    apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_sdag(pos_t pos){
    complex<double> mat[2] = {1., -imag_I};
    apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);

}

template <class real_t>
void StateVector<real_t>::apply_t(pos_t pos){
    complex<double> p = imag_I*PI/4.;
    complex<double> mat[2] = {1., std::exp(p)};
   apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);

}

template <class real_t>
void StateVector<real_t>::apply_tdag(pos_t pos){
    complex<double> p = -imag_I*PI/4.;
    complex<double> mat[2] = {1., std::exp(p)};
    apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);

}

template <class real_t>
void StateVector<real_t>::apply_p(pos_t pos, real_t phase){
    complex<double> p = imag_I*phase;
    complex<double> mat[2] = {1., std::exp(p)};
    apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);
}


template <class real_t>
void StateVector<real_t>::apply_rx(pos_t pos, real_t theta){
    complex<double> mat[4] = {std::cos(theta/2), -imag_I*std::sin(theta/2), -imag_I*std::sin(theta/2), std::cos(theta/2)};
    apply_one_targe_gate_general<1>(vector<pos_t>{pos}, mat);
}


template <class real_t>
void StateVector<real_t>::apply_ry(pos_t pos, real_t theta){
    complex<double> mat[4] = {std::cos(theta/2), -std::sin(theta/2),std::sin(theta/2), std::cos(theta/2)};
    apply_one_targe_gate_real<1>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_rz(pos_t pos, real_t theta){
    complex<double> z0 = -imag_I*theta/2.;
    complex<double> z1 = imag_I*theta/2.;
    complex<double> mat[2] = {std::exp(z0), std::exp(z1)};
    apply_one_targe_gate_diag<1>(vector<pos_t>{pos}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cnot(pos_t control, pos_t targe){
    apply_one_targe_gate_x<2>(vector<pos_t>{control, targe});
}

template <class real_t>
void StateVector<real_t>::apply_cz(pos_t control, pos_t targe){
    complex<double> mat[2] = {1., -1.};
    apply_one_targe_gate_diag<2>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cp(pos_t control, pos_t targe, real_t phase){
    complex<double> p = imag_I*phase;
    complex<double> mat[2] = {1., std::exp(p)};
    apply_one_targe_gate_diag<2>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_crx(pos_t control, pos_t targe,  real_t theta){
    complex<double> mat[4] = {std::cos(theta/2), -imag_I*std::sin(theta/2), -imag_I*std::sin(theta/2), std::cos(theta/2)};
    
    apply_one_targe_gate_general<2>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_cry(pos_t control, pos_t targe,  real_t theta){
     complex<double> mat[4] = {std::cos(theta/2), -std::sin(theta/2),std::sin(theta/2), std::cos(theta/2)};
    
    apply_one_targe_gate_real<2>(vector<pos_t>{control, targe}, mat);
}

template <class real_t>
void StateVector<real_t>::apply_ccx(pos_t control1, pos_t control2, pos_t targe){
    apply_one_targe_gate_x<3>(vector<pos_t>{control1, control2, targe});
}

/////// General implementation /////////

template <class real_t>
template <int num>
void StateVector<real_t>::apply_one_targe_gate_general(vector<pos_t> const& posv, complex<double> *mat){
    
    std::function<size_t(size_t)> getind_func_near;
    std::function<size_t(size_t)> getind_func;
    size_t rsize;
    size_t offset;
    size_t targe;
    size_t control;
    size_t setbit;
    size_t poffset;
    bool has_control=false;
    if (num == 1){
        targe = posv[0];
        offset = 1ll<<targe;
        rsize = size_>>1;
        getind_func_near = [&](size_t j)-> size_t {
            return 2*j;
        };
        
        getind_func = [&](size_t j)-> size_t {
            return (j&(offset-1)) | (j>>targe<<targe<<1);
        };

    }
    else if(num == 2){
        
        has_control = true;
        control = posv[0];
        targe = posv[1];
        offset = 1ll<<targe;
        setbit = 1ll<<control;
        if (control>targe) {
            control--;
        }
        poffset=1ll<<control;
        rsize = size_>>2;
        getind_func = [&](size_t j) -> size_t {
            size_t i = (j>>control<<(control+1))|(j&(poffset-1));
            i = (i>>targe<<(targe+1))|(i&(offset-1))|setbit;
            return i;
        };

        getind_func_near = getind_func;

        
    }
    else if(num == 3){
        has_control = true;
        control = *min_element(posv.begin(), posv.end()-1);
        targe = *(posv.end()-1);
        offset = 1ll<<targe;
        vector<pos_t> posv_sorted = posv;
        sort(posv_sorted.begin(), posv_sorted.end());
        rsize = size_>>posv.size();
        getind_func = [&](size_t j)-> size_t{
            size_t i = j;
            for (size_t k=0;k < posv.size();k++){
                size_t _pos = posv_sorted[k];
                i = (i&((1ll<<_pos)-1)) | (i>>_pos<<_pos<<1);
            }
            for (size_t k=0;k < posv.size()-1;k++){
                i |= 1ll<<posv[k];
            }
            return i;
        };
        getind_func_near = getind_func;   
    }
    
    const complex<real_t> mat00 = mat[0];
    const complex<real_t> mat01 = mat[1];
    const complex<real_t> mat10 = mat[2];
    const complex<real_t> mat11 = mat[3];
    if (targe == 0){            
#pragma omp parallel for
            for(omp_i j = 0;j < rsize;j++){
                size_t i = getind_func_near(j);
                complex<real_t> temp = data_[i];
                data_[i] = mat00*data_[i] + mat01*data_[i+1];
                data_[i+1] = mat10*temp + mat11*data_[i+1];
            }
      
    }else if (has_control && control == 0){ //single step
       
#pragma omp parallel for
            for(omp_i j = 0;j < rsize;j++){
                size_t i = getind_func(j);
                complex<real_t> temp = data_[i];
                data_[i] = mat00*data_[i] + mat01*data_[i+offset];
                data_[i+offset] = mat10*temp + mat11*data_[i+offset];
            }

    }else{//unroll to 2
#ifdef USE_SIMD
    __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(),mat[0].real(),  mat[0].real());
    __m256d m_00im = _mm256_set_pd(mat[0].imag(),  -mat[0].imag(),  mat[0].imag(),  -mat[0].imag());
    __m256d m_01re = _mm256_set_pd(mat[1].real(), mat[1].real(),  mat[1].real(), mat[1].real());
    __m256d m_01im = _mm256_set_pd(mat[1].imag(), -mat[1].imag(),  mat[1].imag(), -mat[1].imag());

    __m256d m_10re = _mm256_set_pd(mat[2].real(), mat[2].real(), mat[2].real(), mat[2].real());
    __m256d m_10im = _mm256_set_pd(mat[2].imag(),  -mat[2].imag(),mat[2].imag(), -mat[2].imag());
    __m256d m_11re = _mm256_set_pd(mat[3].real(), mat[3].real(), mat[3].real(), mat[3].real());
    __m256d m_11im = _mm256_set_pd(mat[3].imag(), -mat[3].imag(), mat[3].imag(),  -mat[3].imag());
#pragma omp parallel for
        for(omp_i j = 0;j < rsize; j+= 2){        
            size_t i = getind_func(j);
            
            double* p0 = (double*)(data_.data()+i);
            double* p1 = (double*)(data_.data()+i+offset);
            //load data
            __m256d data0 = _mm256_loadu_pd(p0); //lre_0, lim_0, rre_0, rim_0
            __m256d data1 = _mm256_loadu_pd(p1); //lre_1, lim_1, rre_1, rim_1
            __m256d data0_p = _mm256_permute_pd(data0, 5);
            __m256d data1_p = _mm256_permute_pd(data1, 5);

             //row0
            __m256d temp00re = _mm256_mul_pd(m_00re, data0);
            __m256d temp00im = _mm256_mul_pd(m_00im, data0_p);
            __m256d temp00 = _mm256_add_pd(temp00re, temp00im);
            __m256d temp01re = _mm256_mul_pd(m_01re, data1);
            __m256d temp01im = _mm256_mul_pd(m_01im, data1_p);
            __m256d temp01 = _mm256_add_pd(temp01re, temp01im);
            __m256d temp0 = _mm256_add_pd(temp00, temp01);

            //row1
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
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = getind_func(j);
            size_t i1 = i+1;
            complex<real_t> temp = data_[i];
            complex<real_t> temp1 = data_[i1];
            data_[i] = mat00*data_[i] + mat01*data_[i+offset];
            data_[i+offset] = mat10*temp + mat11*data_[i+offset];
            data_[i1] = mat00*data_[i1] + mat01*data_[i1+offset];
            data_[i1+offset] = mat10*temp1 + mat11*data_[i1+offset];
        }
#endif
    }
}           


template <class real_t>
template <int num>
void StateVector<real_t>::apply_one_targe_gate_x(vector<pos_t> const& posv){
    std::function<size_t(size_t)> getind_func_near;
    std::function<size_t(size_t)> getind_func;
    size_t rsize;
    size_t offset;
    size_t targe;
    size_t control;
    size_t setbit;
    size_t poffset;
    vector<pos_t> posv_sorted = posv;
    bool has_control=false;
    if (num == 1){
        targe = posv[0];
        offset = 1ll<<targe;
        rsize = size_>>1;
        getind_func_near = [&](size_t j)-> size_t {
            return 2*j;
        };
        
        getind_func = [&](size_t j)-> size_t {
            return (j&(offset-1)) | (j>>targe<<targe<<1);
        };

    }
    else if(num == 2){
        has_control = true;
        control = posv[0];
        targe = posv[1];
        offset = 1ll<<targe;
        setbit = 1ll<<control;
        if (control>targe) {
            control--;
        }
        poffset=1ll<<control;
        rsize = size_>>2;
        getind_func = [&](size_t j) -> size_t {
            size_t i = (j>>control<<(control+1))|(j&(poffset-1));
            i = (i>>targe<<(targe+1))|(i&(offset-1))|setbit;
            return i;
        };
        getind_func_near = getind_func;
    }
    else if(num == 3){
        has_control = true;
        control = *min_element(posv.begin(), posv.end()-1);
        targe = *(posv.end()-1);
        offset = 1ll<<targe;
        sort(posv_sorted.begin(), posv_sorted.end());
        rsize = size_>>posv.size();

        getind_func = [&](size_t j) -> size_t{
            size_t i = j;
            for (size_t k=0;k < posv.size();k++){
                size_t _pos = posv_sorted[k];
                i = (i&((1ll<<_pos)-1)) | (i>>_pos<<_pos<<1);
            }
            for (size_t k=0;k < posv.size()-1;k++){
                i |= 1ll<<posv[k];
            }
            return i;
        };
        getind_func_near = getind_func;   
    }
  
    if (targe == 0){
#ifdef USE_SIMD            
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j++){
            size_t i = getind_func_near(j);
            double* ptr = (double*)(data_.data() + i);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data, 78); 
            _mm256_storeu_pd(ptr, data);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j++){
            size_t i = getind_func(j);
            std::swap(data_[i], data_[i+1]);
        }
#endif  
    }else if (has_control && control == 0){ //single step
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j++){
            size_t i = getind_func(j);
            std::swap(data_[i], data_[i+offset]);
        }

    }else{//unroll to 2
#ifdef USE_SIMD
#pragma omp parallel for
        for(omp_i j = 0;j < rsize; j+= 2){        
            size_t i = getind_func(j);
            double* ptr0 = (double*)(data_.data() + i);
            double* ptr1 = (double*)(data_.data() + i + offset);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = getind_func(j);
            size_t i1 = i+1;
            std::swap(data_[i], data_[i+offset]);
            std::swap(data_[i1], data_[i1+offset]);
        }
#endif
    }
}           

template <class real_t>
template <int num>
void StateVector<real_t>::apply_one_targe_gate_real(vector<pos_t> const& posv, complex<double> *mat){
    
    std::function<size_t(size_t)> getind_func_near;
    std::function<size_t(size_t)> getind_func;
    size_t rsize;
    size_t offset;
    size_t targe;
    size_t control;
    size_t setbit;
    size_t poffset;
    bool has_control=false;
    if (num == 1){
        targe = posv[0];
        offset = 1ll<<targe;
        rsize = size_>>1;
        getind_func_near = [&](size_t j)-> size_t {
            return 2*j;
        };
        
        getind_func = [&](size_t j)-> size_t {
            return (j&(offset-1)) | (j>>targe<<targe<<1);
        };

    }
    else if(num == 2){
        
        has_control = true;
        control = posv[0];
        targe = posv[1];
        offset = 1ll<<targe;
        setbit = 1ll<<control;
        if (control>targe) {
            control--;
        }
        poffset=1ll<<control;
        rsize = size_>>2;
        getind_func = [&](size_t j) -> size_t {
            size_t i = (j>>control<<(control+1))|(j&(poffset-1));
            i = (i>>targe<<(targe+1))|(i&(offset-1))|setbit;
            return i;
        };

        getind_func_near = getind_func;
    }
    else if(num == 3){
        has_control = true;
        control = *min_element(posv.begin(), posv.end()-1);
        targe = *(posv.end()-1);
        offset = 1ll<<targe;
        vector<pos_t> posv_sorted = posv;
        sort(posv_sorted.begin(), posv_sorted.end());
        rsize = size_>>posv.size();
        getind_func = [&](size_t j)-> size_t{
            size_t i = j;
            for (size_t k=0;k < posv.size();k++){
                size_t _pos = posv_sorted[k];
                i = (i&((1ll<<_pos)-1)) | (i>>_pos<<_pos<<1);
            }
            for (size_t k=0;k < posv.size()-1;k++){
                i |= 1ll<<posv[k];
            }
            return i;
        };
        getind_func_near = getind_func;   
    }
    
    const double mat00 = mat[0].real();
    const double mat01 = mat[1].real();
    const double mat10 = mat[2].real();
    const double mat11 = mat[3].real();
    if (targe == 0){            
#pragma omp parallel for
            for(omp_i j = 0;j < rsize;j++){
                size_t i = getind_func_near(j);
                complex<real_t> temp = data_[i];
                data_[i] = mat00*data_[i] + mat01*data_[i+1];
                data_[i+1] = mat10*temp + mat11*data_[i+1];
            }
      
    }else if (has_control && control == 0){ //single step
       
#pragma omp parallel for
            for(omp_i j = 0;j < rsize;j++){
                size_t i = getind_func(j);
                complex<real_t> temp = data_[i];
                data_[i] = mat00*data_[i] + mat01*data_[i+offset];
                data_[i+offset] = mat10*temp + mat11*data_[i+offset];
            }
    }else{//unroll to 2
#ifdef USE_SIMD
    __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(),mat[0].real(),  mat[0].real());
    __m256d m_01re = _mm256_set_pd(mat[1].real(), mat[1].real(),  mat[1].real(), mat[1].real());
    __m256d m_10re = _mm256_set_pd(mat[2].real(), mat[2].real(), mat[2].real(), mat[2].real());
    __m256d m_11re = _mm256_set_pd(mat[3].real(), mat[3].real(), mat[3].real(), mat[3].real());
#pragma omp parallel for
        for(omp_i j = 0;j < rsize; j+= 2){        
            size_t i = getind_func(j);
            
            double* p0 = (double*)(data_.data()+i);
            double* p1 = (double*)(data_.data()+i+offset);
             //load data
            __m256d data0 = _mm256_loadu_pd(p0); //lre_0, lim_0, rre_0, rim_0
            __m256d data1 = _mm256_loadu_pd(p1); //lre_1, lim_1, rre_1, rim_1
            __m256d data0_p = _mm256_permute_pd(data0, 5);
            __m256d data1_p = _mm256_permute_pd(data1, 5);

                //row0
            __m256d temp00re = _mm256_mul_pd(m_00re, data0);
            __m256d temp01re = _mm256_mul_pd(m_01re, data1);
            __m256d temp0 = _mm256_add_pd(temp00re, temp01re);

            //row1
            __m256d temp10re = _mm256_mul_pd(m_10re, data0);
            __m256d temp11re = _mm256_mul_pd(m_11re, data1);
            __m256d temp1 = _mm256_add_pd(temp10re, temp11re);

            _mm256_storeu_pd(p0, temp0);
            _mm256_storeu_pd(p1, temp1);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = getind_func(j);
            size_t i1 = i+1;
            complex<real_t> temp = data_[i];
            complex<real_t> temp1 = data_[i1];
            data_[i] = mat00*data_[i] + mat01*data_[i+offset];
            data_[i+offset] = mat10*temp + mat11*data_[i+offset];
            data_[i1] = mat00*data_[i1] + mat01*data_[i1+offset];
            data_[i1+offset] = mat10*temp1 + mat11*data_[i1+offset];
        }
#endif
    }
}      


template <class real_t>
template <int num>
void StateVector<real_t>::apply_one_targe_gate_diag(vector<pos_t> const& posv, complex<double> *mat){
    std::function<size_t(size_t)> getind_func_near;
    std::function<size_t(size_t)> getind_func;
    size_t rsize;
    size_t offset;
    size_t targe;
    size_t control;
    size_t setbit;
    size_t poffset;
    bool has_control=false;
    if (num == 1){
        targe = posv[0];
        offset = 1ll<<targe;
        rsize = size_>>1;
        getind_func_near = [&](size_t j)-> size_t {
            return 2*j;
        };
        
        getind_func = [&](size_t j)-> size_t {
            return (j&(offset-1)) | (j>>targe<<targe<<1);
        };

    }
    else if(num == 2){
        
        has_control = true;
        control = posv[0];
        targe = posv[1];
        offset = 1ll<<targe;
        setbit = 1ll<<control;
        if (control>targe) {
            control--;
        }
        poffset=1ll<<control;
        rsize = size_>>2;
        getind_func = [&](size_t j) -> size_t {
            size_t i = (j>>control<<(control+1))|(j&(poffset-1));
            i = (i>>targe<<(targe+1))|(i&(offset-1))|setbit;
            return i;
        };

        getind_func_near = getind_func;
    
    }
    else if(num == 3){
        has_control = true;
        control = *min_element(posv.begin(), posv.end()-1);
        targe = *(posv.end()-1);
        offset = 1ll<<targe;
        vector<pos_t> posv_sorted = posv;
        sort(posv_sorted.begin(), posv_sorted.end());
        rsize = size_>>posv.size();
        getind_func = [&](size_t j)-> size_t{
            size_t i = j;
            for (size_t k=0;k < posv.size();k++){
                size_t _pos = posv_sorted[k];
                i = (i&((1ll<<_pos)-1)) | (i>>_pos<<_pos<<1);
            }
            for (size_t k=0;k < posv.size()-1;k++){
                i |= 1ll<<posv[k];
            }
            return i;
        };
        getind_func_near = getind_func;   
    }
    
    if (targe == 0){            
#pragma omp parallel for
            for(omp_i j = 0;j < rsize;j++){
                size_t i = getind_func_near(j);
                data_[i] *= mat[0];
                data_[i+1] *= mat[1];
            }
      
    }else if (has_control && control == 0){ //single step
       
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j++){
            size_t i = getind_func(j);
            complex<real_t> temp = data_[i];
            data_[i] *= mat[0];
            data_[i+offset] *= mat[1];
        }

    }else{//unroll to 2
#ifdef USE_SIMD
     __m256d m_00re = _mm256_set_pd(mat[0].real(), mat[0].real(),mat[0].real(),  mat[0].real());
    __m256d m_00im = _mm256_set_pd(mat[0].imag(),  -mat[0].imag(),  mat[0].imag(),  -mat[0].imag());
    __m256d m_11re = _mm256_set_pd(mat[1].real(), mat[1].real(),  mat[1].real(), mat[1].real());
    __m256d m_11im = _mm256_set_pd(mat[1].imag(), -mat[1].imag(),  mat[1].imag(), -mat[1].imag());
#pragma omp parallel for
        for(omp_i j = 0;j < rsize; j+= 2){        
            size_t i = getind_func(j);
            
            double* p0 = (double*)(data_.data()+i);
            double* p1 = (double*)(data_.data()+i+offset);

            //load data
            __m256d data0 = _mm256_loadu_pd(p0); //lre_0, lim_0, rre_0, rim_0
            __m256d data1 = _mm256_loadu_pd(p1); //lre_1, lim_1, rre_1, rim_1
            __m256d data0_p = _mm256_permute_pd(data0, 5);
            __m256d data1_p = _mm256_permute_pd(data1, 5);

             //row0
            __m256d temp00re = _mm256_mul_pd(m_00re, data0);
            __m256d temp00im = _mm256_mul_pd(m_00im, data0_p);
            __m256d temp00 = _mm256_add_pd(temp00re, temp00im);

            //row1
            __m256d temp11re = _mm256_mul_pd(m_11re, data1);
            __m256d temp11im = _mm256_mul_pd(m_11im, data1_p);
            __m256d temp11 = _mm256_add_pd(temp11re, temp11im);

            _mm256_storeu_pd(p0, temp00);
            _mm256_storeu_pd(p1, temp11);
        }
#else
#pragma omp parallel for
        for(omp_i j = 0;j < rsize;j += 2){
            size_t i = getind_func(j);
            size_t i1 = i+1;
            data_[i] *= mat[0];
            data_[i+offset] *= mat[1];
            data_[i1] *= mat[0];
            data_[i1+offset] *= mat[1];
        }
#endif
    }
}      


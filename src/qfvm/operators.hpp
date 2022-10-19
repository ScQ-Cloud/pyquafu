#pragma once

#include "types.hpp"
#include <iostream>
#include "util.h"

class QuantumOperator{
    private:
        vector<pos_t> positions_;
        uint control_num_;
        uint targe_num_;
        string name_;
        bool diag_;
        bool real_;
        vector<complex<double>> matv_;
    public:
        QuantumOperator();
        QuantumOperator(string name, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, vector<complex<double>> const &matv, bool diag=false, bool real=false);
        QuantumOperator(string name, vector<pos_t> const &positions, uint control_num, vector<complex<double>> const &matv, bool diag=false, bool real=false);

        string name() const {return name_;}
        bool has_control() const{return control_num_ == 0 ? false : true;}
        bool is_real() const{ return real_; }
        bool is_diag() const{ return diag_; }
        vector<complex<double>> get_matrix() const { return matv_;}
        uint control_num() const { return control_num_; } 
        uint targe_num() const { return targe_num_; }
        vector<pos_t> positions(){ return positions_; }

        virtual void apply_to_state(StateVector<double> & state){ };
};


QuantumOperator::QuantumOperator(string name, vector<pos_t> const &positions, uint control_num, vector<complex<double>> const &matv, bool diag, bool real)
:
name_(name),
positions_(positions),
control_num_(control_num),
diag_(diag),
real_(real),
matv_(matv){ }


QuantumOperator::QuantumOperator(){ };
QuantumOperator::QuantumOperator(string name, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, vector<complex<double>> const &matv, bool diag, bool real)
:
name_(name),
diag_(diag),
real_(real),
matv_(matv){
    positions_ = control_qubits;
    positions_.insert(positions_.end(), targe_qubits.begin(), targe_qubits.end());
    control_num_ = control_qubits.size();
    targe_num_ = targe_qubits.size();
}

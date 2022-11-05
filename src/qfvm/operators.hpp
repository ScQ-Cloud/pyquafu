#pragma once

#include <iostream>
#include "statevector.hpp"

class QuantumOperator{
    protected:
        string name_;
        vector<pos_t> positions_;
        vector<double> paras_;
        uint control_num_;
        uint targe_num_;  
        bool diag_;
        bool real_;
        RowMatrixXcd mat_;
    public:
        //Constructor
        QuantumOperator();
        QuantumOperator(string name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat, bool diag=false, bool real=false);
        QuantumOperator(string name,vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat, bool diag=false, bool real=false);
        // virtual ~QuantumOperator();

        //data accessor
        string name() const {return name_;}
        vector<double> paras() const {return paras_;}
        bool has_control() const{return control_num_ == 0 ? false : true;}
        bool is_real() const{ return real_; }
        bool is_diag() const{ return diag_; }
        RowMatrixXcd mat() const { return mat_;}
        uint control_num() const { return control_num_; } 
        uint targe_num() const { return targe_num_; }
        vector<pos_t> positions(){ return positions_; }
        explicit operator bool() const {
            return !(name_ == "empty");
        }
        //Apply method
        virtual void apply_to_state(StateVector<double> & state){ };
};


QuantumOperator::QuantumOperator() : name_("empty"){ };
QuantumOperator::QuantumOperator(string name, vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat, bool diag, bool real)
:
name_(name),
paras_(paras),
positions_(positions),
control_num_(control_num),
targe_num_(positions.size()-control_num),
diag_(diag),
real_(real),
mat_(mat){ }

QuantumOperator::QuantumOperator(string name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat, bool diag, bool real)
:
name_(name),
paras_(paras),
diag_(diag),
real_(real),
mat_(mat){
    positions_ = control_qubits;
    positions_.insert(positions_.end(), targe_qubits.begin(), targe_qubits.end());
    control_num_ = control_qubits.size();
    targe_num_ = targe_qubits.size();
}

#pragma once

#include "statevector.hpp"
#include "qasm.hpp"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include "types.hpp"
#include "util.h"

namespace py = pybind11;
using namespace pybind11::literals;

class QuantumOperator;

class Instruction {
protected:
    string name_ = "empty";
    vector<pos_t> positions_;
  
public:
    Instruction(){ };
    Instruction(string const&name, vector<pos_t> const& positions):
    name_(name),
    positions_(positions) { }
    string name() const { return name_; }
    vector<pos_t> positions() const { return positions_; }
    explicit operator bool() const {
            return !(name_ == "empty");
        }
    
    //interface
    virtual vector<double> paras() const {return {};}
    virtual bool has_control() const{ return false; }
    virtual bool is_real() const{ return false; }
    virtual bool is_diag() const{ return false; }
    virtual RowMatrixXcd targ_mat() const { return RowMatrixXcd(0,0);}
    virtual RowMatrixXcd full_mat() const { return RowMatrixXcd(0,0); }
    virtual uint control_num() const { return 0; } 
    virtual uint targe_num() const { return 0; }
 
    virtual vector<pos_t> qbits() const { return {}; }
    virtual vector<pos_t> cbits() const { return {}; }
 
    virtual uint condition() const { return 0; }
    virtual vector<std::unique_ptr<Instruction>>& instructions() {  vector<std::unique_ptr<Instruction>> empty = {};
    return empty; }

};


class QuantumOperator :  public Instruction {
protected:
    vector<double> paras_;
    uint control_num_;
    uint targe_num_;
    bool diag_;
    bool real_;
    RowMatrixXcd targ_mat_;
    RowMatrixXcd full_mat_;

   
public:
    //Constructor
    QuantumOperator() { };
    QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat = RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0), bool diag=false, bool real=false);

    QuantumOperator(string const name,vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat=RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0), bool diag=false, bool real=false);

    QuantumOperator(string const name, vector<pos_t> const &positions, RowMatrixXcd const &mat=RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0));
    // virtual ~QuantumOperator();

    //data accessor
    virtual vector<double> paras() const override {return paras_;}
    virtual bool has_control() const override {return control_num_ == 0 ? false : true;}
    virtual bool is_real() const override { return real_; }
    virtual bool is_diag() const override { return diag_; }
    virtual RowMatrixXcd targ_mat() const override { return targ_mat_;}
    virtual RowMatrixXcd full_mat() const override {return full_mat_;}
    virtual uint control_num() const override { return control_num_; } 
    virtual uint targe_num() const override { return targe_num_; }
};

QuantumOperator::QuantumOperator(string const name, vector<pos_t> const &positions, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat)
:
Instruction(name, positions),
paras_(vector<double>(0)),
control_num_(-1),
targe_num_(0),
diag_(false),
real_(false),
targ_mat_(mat),
full_mat_(full_mat){ }

QuantumOperator::QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat, bool diag, bool real)
:
Instruction(name, positions),
paras_(paras),
control_num_(control_num),
targe_num_(positions.size()-control_num),
diag_(diag),
real_(real),
targ_mat_(mat), 
full_mat_(full_mat){ }

QuantumOperator::QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat, bool diag, bool real)
:
paras_(paras),
diag_(diag),
real_(real),
targ_mat_(mat),
full_mat_(full_mat){
    Instruction::name_ = name;
    Instruction::positions_ = control_qubits;
    Instruction::positions_.insert(Instruction::positions_.end(), targe_qubits.begin(), targe_qubits.end());
    control_num_ = control_qubits.size();
    targe_num_ = targe_qubits.size();
}

class Measures : public Instruction {
protected:
      vector<pos_t> qbits_;
      vector<pos_t> cbits_;
      
public:
      Measures(vector<pos_t> &qbits, vector<pos_t> &cbits) :
      qbits_(qbits),
      cbits_(cbits)
      { 
        Instruction:name_ = "measure";
      }
      virtual vector<pos_t> qbits() const override { return qbits_; }
      virtual vector<pos_t> cbits() const override { return cbits_; }
};


class Reset : public Instruction{
protected:
    vector<pos_t> qbits_;
public:
    Reset(vector<pos_t> const& qbits) :
    qbits_(qbits)
    {  
        Instruction:name_ = "reset";
    }
    virtual vector<pos_t> qbits() const override { return qbits_; }
};

class Cif : public Instruction{
protected:
    vector<pos_t> cbits_;
    uint condition_;
    vector<std::unique_ptr<Instruction>> instructions_;
public:
    Cif(vector<pos_t> &cbits, uint condition, vector<std::unique_ptr<Instruction>> &instructions)
    :
    cbits_(cbits), 
    condition_(condition),
    instructions_(std::move(instructions))
    {  
        Instruction:name_ = "cif";
    }
    virtual uint condition() const override{ return condition_; }
    virtual vector<std::unique_ptr<Instruction>>& instructions() override { return instructions_; }
    virtual vector<pos_t> cbits() const override { return cbits_; }
};


// Construct C++ operators from pygates
std::unique_ptr<Instruction> from_pyops(py::object const& obj, bool get_full_mat=false, bool reverse=true) {
    string name;
    vector<pos_t> positions;
    vector<pos_t> qbits;
    vector<pos_t> cbits;
    vector<double> paras;
    uint control_num = 0;
    RowMatrixXcd targ_mat;
    RowMatrixXcd full_mat;

    name = obj.attr("name").attr("lower")().cast<string>();
    if (!(name == "barrier" || name == "delay" || name == "id" ||
        name == "measure" || name == "reset" || name == "cif")) {
    //QuantumGate
        positions = obj.attr("pos").cast<vector<pos_t>>();
        paras = obj.attr("_paras").cast<vector<double>>();

        if (py::hasattr(obj, "ctrls")) {
        control_num = py::len(obj.attr("ctrls"));
        }

        if (OPMAP.count(name) == 0){
            if (py::hasattr(obj, "_targ_matrix")){
                targ_mat = obj.attr("_get_targ_matrix")("reverse_order"_a=true).cast<RowMatrixXcd>();
            }else{ //Single gate
                targ_mat = obj.attr("matrix").cast<RowMatrixXcd>();
            }
        }
        else{
            targ_mat = RowMatrixXcd(0, 0);
        }

        if (get_full_mat){
            full_mat = obj.attr("_get_raw_matrix")("reverse_order"_a=reverse).cast<RowMatrixXcd>();
            }else{
            full_mat = RowMatrixXcd(0, 0);
        }
        return std::make_unique<QuantumOperator>(name, paras, positions, control_num, std::move(targ_mat), std::move(full_mat));

    } else if (name == "measure") {
        qbits = obj.attr("qbits").cast<vector<pos_t>>();
        cbits = obj.attr("cbits").cast<vector<pos_t>>();

        return std::make_unique<Measures>(qbits, cbits);

    } else if (name == "reset") {
        positions = obj.attr("pos").cast<vector<pos_t>>();
        return  std::make_unique<Reset>(positions);

    } else if (name == "cif") {
        uint condition = 0;
        vector<std::unique_ptr<Instruction>> instructions;
        cbits = obj.attr("cbits").cast<vector<pos_t>>();
        condition = obj.attr("condition").cast<pos_t>();

        // Recursively handdle instruction
        if (py::isinstance<py::list>(obj.attr("instructions"))) {
            auto pyops = obj.attr("instructions");
            for (auto pyop_h : pyops) {
                py::object pyop = py::reinterpret_borrow<py::object>(pyop_h);
                std::unique_ptr<Instruction> ins = from_pyops(pyop);
                if (*ins) 
                    instructions.push_back(std::move(ins));
            }   
        }
        return  std::make_unique<Cif>(cbits, condition, instructions);

    } else {
        return  std::make_unique<QuantumOperator>();
    }
}

void check_operator(Instruction &op){
    std::cout << "-------------" << std::endl;

    std::cout << "name: " << op.name() << std::endl;
    std::cout << "pos: ";
    Qfutil::printVector(op.positions());

    std::cout << "paras: ";
    Qfutil::printVector(op.paras());

    std::cout << "control number: ";
    std::cout << op.control_num() << std::endl;

    std::cout << "matrix: " << std::endl;
    std::cout << op.targ_mat() << std::endl;
    std::cout << "flatten matrix: " << std::endl;
    auto mat = op.targ_mat();
    // Eigen::Map<Eigen::RowVectorXcd> v1(mat.data(), mat.size());
    // std::cout << "v1: " << v1 << std::endl;
    auto matv = mat.data();
    for (auto i = 0;i < mat.size();i++){
        std::cout << matv[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "full matrix: " << std::endl;
    std::cout << op.full_mat() << std::endl;
    std::cout << "-------------" << std::endl;
}
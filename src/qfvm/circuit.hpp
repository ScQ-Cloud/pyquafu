#pragma once
#include "operators.hpp"
#include "qasm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "util.h"
#include <Eigen/Core>
#include <algorithm>

namespace py = pybind11;
using namespace pybind11::literals;

// Construct C++ operators from pygates
QuantumOperator from_pyops(py::object const &obj){
    string name;
    vector<pos_t> positions;
    vector<double> paras;
    uint control_num = 0;
    RowMatrixXcd mat;
    
    name = obj.attr("name").attr("lower")().cast<string>();
    if (!(name == "barrier" || name == "delay" || name == "id"))
    {
        if (py::isinstance<py::list>(obj.attr("pos"))){
            positions = obj.attr("pos").cast<vector<pos_t>>();
        }
        else if(py::isinstance<py::int_>(obj.attr("pos"))){
            positions = vector<pos_t>{obj.attr("pos").cast<pos_t>()};
        }

        if (py::isinstance<py::list>(obj.attr("paras"))){
            paras = obj.attr("paras").cast<vector<double>>();
        }
        else if(py::isinstance<py::float_>(obj.attr("paras")) || py::isinstance<py::int_>(obj.attr("paras"))){
            paras = vector<double>{obj.attr("paras").cast<double>()};
        }

        if (py::hasattr(obj, "ctrls")){
                control_num = py::len(obj.attr("ctrls"));
        }
        
        //Reverse order for multi-target gate
        if (py::hasattr(obj, "_targ_matrix")){
                mat = obj.attr("get_targ_matrix")("reverse_order"_a=true).cast<RowMatrixXcd>();
        }
        else{ //Single target gate
                mat = obj.attr("matrix").cast<RowMatrixXcd>();
        }
        return QuantumOperator(name, paras, positions, control_num, mat);
    }
    else{
        return QuantumOperator();
    }
   
}

void check_operator(QuantumOperator &op){
    std::cout << "-------------" << std::endl;

    std::cout << "name: " << op.name() << std::endl;
    std::cout << "pos: ";
    Qfutil::printVector(op.positions());

    std::cout << "paras: ";
    Qfutil::printVector(op.paras());

    std::cout << "control number: ";
    std::cout << op.control_num() << std::endl;

    std::cout << "matrix: " << std::endl;
    std::cout << op.mat() << std::endl;

    std::cout << "flatten matrix: " << std::endl;
    auto mat = op.mat();
    // Eigen::Map<Eigen::RowVectorXcd> v1(mat.data(), mat.size());
    // std::cout << "v1: " << v1 << std::endl;
    auto matv = mat.data();
    for (auto i = 0;i < mat.size();i++){
        std::cout << matv[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-------------" << std::endl;
}


class Circuit{
    private:
        uint qubit_num_;    
        vector<QuantumOperator> gates_{0};

    public:
    Circuit();
    explicit Circuit(uint qubit_num);
    explicit Circuit(vector<QuantumOperator> &gates);
    explicit Circuit(py::object const&pycircuit); 

    void add_gate(QuantumOperator &gate);
    void compress_gates();
    uint qubit_num() const { return qubit_num_; }
    vector<QuantumOperator>gates() const { return gates_; }

};

void Circuit::add_gate(QuantumOperator &gate){
    for (pos_t pos : gate.positions()){
        if (pos > qubit_num_) {
            throw "invalid position on quantum registers";
        }
        else{
            gates_.push_back(gate);
        }
    }
}

 Circuit::Circuit(){};
 Circuit::Circuit(uint qubit_num)
 :
 qubit_num_(qubit_num){ }

 Circuit::Circuit(vector<QuantumOperator> &gates)
 :
 gates_(gates){
    qubit_num_ = 0;
    for (auto gate : gates){
        for (pos_t pos : gate.positions()){
            if (pos+1 > qubit_num_){ qubit_num_ = pos+1; }
        }
    }
}

Circuit::Circuit(py::object const&pycircuit)
{
    auto pygates = pycircuit.attr("gates");
    auto used_qubits = pycircuit.attr("used_qubits").cast<vector<pos_t>>();
    qubit_num_ = *std::max_element(used_qubits.begin(), used_qubits.end())+1;
    for (auto pygate_h : pygates){
        py::object pygate = py::reinterpret_borrow<py::object>(pygate_h);
        QuantumOperator gate = from_pyops(pygate);
        if (gate){
            gates_.push_back(std::move(gate));
        }        
    }
} 

void Circuit::compress_gates(){}

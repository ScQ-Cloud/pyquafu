#pragma once
#include "instructions.hpp"
#include "qasm.hpp"
#include "util.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>


namespace py = pybind11;
using namespace pybind11::literals;


class Circuit {
    private:
        uint qubit_num_;
        vector<std::unique_ptr<Instruction>> instructions_;
        uint max_targe_num_;
        uint cbit_num_;
        // to sample count
        vector<std::pair<uint, uint>> measure_vec_;
        bool final_measure_ = true;

    public:
        Circuit();
        explicit Circuit(uint qubit_num);
        explicit Circuit(vector<std::unique_ptr<Instruction>> & ops);
        Circuit(py::object const&pycircuit, bool get_full_mat=false, bool reverse=true);

        void add_op(std::unique_ptr<Instruction> op);
        void compress_instructions();
        uint qubit_num() const { return qubit_num_; }
        uint cbit_num() const { return cbit_num_; }
        uint max_targe_num() const { return max_targe_num_; }
        bool final_measure() const { return final_measure_; }
        vector<QuantumOperator> gates();
        vector<std::pair<uint, uint>> measure_vec() { return measure_vec_; }
        vector<std::unique_ptr<Instruction>>& instructions() { return instructions_; }
};

void Circuit::add_op(std::unique_ptr<Instruction> op) {
  for (pos_t pos : op->positions()) {
    if (pos > qubit_num_) {
      throw "invalid position on quantum registers";
    } else {
      instructions_.push_back(std::move(op));
    }
  }
}

Circuit::Circuit(){};
Circuit::Circuit(uint qubit_num) : qubit_num_(qubit_num) {}

Circuit::Circuit(vector<std::unique_ptr<Instruction>>& ops)
    : instructions_(std::move(ops)), max_targe_num_(0) {
  qubit_num_ = 0;
  for (auto& op : instructions_) {
    for (pos_t pos : op->positions()) {
      if (op->targe_num() > max_targe_num_)
        max_targe_num_ = op->targe_num();
      if (pos + 1 > qubit_num_) {
        qubit_num_ = pos + 1;
      }
    }
  }
}

vector<QuantumOperator> Circuit::gates() {
  // provide gates for gpu and custate
  std::vector<std::string> classics = {"measure", "cif", "reset"};
  vector<QuantumOperator> gates;
  for (auto& op : instructions_) {
    if (std::find(classics.begin(), classics.end(), op->name()) ==
        classics.end()) {
        QuantumOperator *gate_ptr = dynamic_cast<QuantumOperator*>(op.get());
        if (gate_ptr != nullptr){
            QuantumOperator gate  = *std::move(gate_ptr); //copy.may use shared_ptr  
            gates.push_back(gate);
        }
        else{
            std::cout << "Dynamic cast failed." << std::endl;
        }
    }
  }
  return gates;
}


Circuit::Circuit(py::object const& pycircuit, bool get_full_mat, bool reverse) : max_targe_num_(0) {
  // auto pygates = pycircuit.attr("gates");
  auto pyops = pycircuit.attr("instructions"); 
  qubit_num_ = pycircuit.attr("num").cast<uint>(); //To consist with other simulators (e.g. qiskit)
  // auto used_qubits = pycircuit.attr("used_qubits").cast<vector<pos_t>>();
// qubit_num_ = *std::max_element(used_qubits.begin(), used_qubits.end()) + 1;

  cbit_num_ = pycircuit.attr("cbits_num").cast<uint>();
  // judge wheather op qubit after measure
  bool measured = false;
  for (auto pyop_h : pyops) {
    py::object pyop = py::reinterpret_borrow<py::object>(pyop_h);
    if (py::hasattr(pyop, "circuit")){ //handle oracle
        auto wrap_circuit = Circuit(pyop.attr("circuit"), get_full_mat, reverse);
        for (auto& op : wrap_circuit.instructions()){
           instructions_.push_back(std::move(op));
        }
    }
    else{
      std::unique_ptr<Instruction> ins = from_pyops(pyop, get_full_mat, reverse);
      check_operator(*(ins));
      if (*ins){
        if (ins->name() == "measure"){
            measured = true;
            // record qbit-cbit measure map
            for (uint i = 0; i < ins->qbits().size(); i++) {
                measure_vec_.push_back(std::make_pair(ins->qbits()[i], ins->cbits()[i]));
            }
        }
        else{
            if (ins->targe_num() > max_targe_num_)
                max_targe_num_ = ins->targe_num();
            if (measured == true)
                final_measure_ = false;
        }
        instructions_.push_back(std::move(ins));
      }
    }
  }
}

void Circuit::compress_instructions() {}

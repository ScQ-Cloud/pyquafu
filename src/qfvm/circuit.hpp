#pragma once
#include "operators.hpp"
#include "qasm.hpp"
#include "util.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

void check_operator(QuantumOperator& op) {
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
  for (auto i = 0; i < mat.size(); i++) {
    std::cout << matv[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "-------------" << std::endl;
}

class Circuit {
private:
  uint qubit_num_;
  vector<QuantumOperator> instructions_;
  uint max_targe_num_;
  uint cbit_num_;
  // to sample count
  vector<std::pair<uint, uint>> measure_vec_;
  bool final_measure_ = true;

public:
  Circuit();
  explicit Circuit(uint qubit_num);
  explicit Circuit(vector<QuantumOperator>& ops);
  explicit Circuit(py::object const& pycircuit);

  void add_op(QuantumOperator& op);
  void compress_instructions();
  uint qubit_num() const { return qubit_num_; }
  uint cbit_num() const { return cbit_num_; }
  uint max_targe_num() const { return max_targe_num_; }
  bool final_measure() const { return final_measure_; }
  vector<QuantumOperator> gates();
  vector<std::pair<uint, uint>> measure_vec() { return measure_vec_; }
  vector<QuantumOperator> instructions() const { return instructions_; }
  QuantumOperator from_pyops(py::object const& obj);
};

void Circuit::add_op(QuantumOperator& op) {
  for (pos_t pos : op.positions()) {
    if (pos > qubit_num_) {
      throw "invalid position on quantum registers";
    } else {
      instructions_.push_back(op);
    }
  }
}

Circuit::Circuit(){};
Circuit::Circuit(uint qubit_num) : qubit_num_(qubit_num) {}

Circuit::Circuit(vector<QuantumOperator>& ops)
    : instructions_(ops), max_targe_num_(0) {
  qubit_num_ = 0;
  for (auto op : ops) {
    for (pos_t pos : op.positions()) {
      if (op.targe_num() > max_targe_num_)
        max_targe_num_ = op.targe_num();
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
  for (auto op : instructions_) {
    if (std::find(classics.begin(), classics.end(), op.name()) ==
        classics.end()) {
      gates.push_back(op);
    }
  }
  return gates;
}

// Construct C++ operators from pygates
QuantumOperator Circuit::from_pyops(py::object const& obj) {
  string name;
  vector<pos_t> positions;
  vector<pos_t> qbits;
  vector<pos_t> cbits;
  vector<double> paras;
  uint control_num = 0;
  RowMatrixXcd mat;

  name = obj.attr("name").attr("lower")().cast<string>();
  if (!(name == "barrier" || name == "delay" || name == "id" ||
        name == "measure" || name == "reset" || name == "cif")) {
    if (py::isinstance<py::list>(obj.attr("pos"))) {
      positions = obj.attr("pos").cast<vector<pos_t>>();
    } else if (py::isinstance<py::int_>(obj.attr("pos"))) {
      positions = vector<pos_t>{obj.attr("pos").cast<pos_t>()};
    }

    if (py::isinstance<py::list>(obj.attr("paras"))) {
      paras = obj.attr("paras").cast<vector<double>>();
    } else if (py::isinstance<py::float_>(obj.attr("paras")) ||
               py::isinstance<py::int_>(obj.attr("paras"))) {
      paras = vector<double>{obj.attr("paras").cast<double>()};
    }

    if (py::hasattr(obj, "ctrls")) {
      control_num = py::len(obj.attr("ctrls"));
    }

    // Reverse order for multi-target gate
    if (py::hasattr(obj, "_targ_matrix")) {
      mat = obj.attr("get_targ_matrix")("reverse_order"_a = true)
                .cast<RowMatrixXcd>();
    } else { // Single target gate
      mat = obj.attr("matrix").cast<RowMatrixXcd>();
    }
    return QuantumOperator(name, paras, positions, control_num, mat);

  } else if (name == "measure") {
    if (py::isinstance<py::list>(obj.attr("qbits"))) {
      qbits = obj.attr("qbits").cast<vector<pos_t>>();
    } else if (py::isinstance<py::int_>(obj.attr("qbits"))) {
      qbits = vector<pos_t>{obj.attr("qbits").cast<pos_t>()};
    }

    if (py::isinstance<py::list>(obj.attr("cbits"))) {
      cbits = obj.attr("cbits").cast<vector<pos_t>>();
    } else if (py::isinstance<py::int_>(obj.attr("cbits"))) {
      cbits = vector<pos_t>{obj.attr("cbits").cast<pos_t>()};
    }
    // record qbit-cbit measure map
    for (uint i = 0; i < qbits.size(); i++) {
      measure_vec_.push_back(std::make_pair(qbits[i], cbits[i]));
    }
    return QuantumOperator(name, qbits, cbits);

  } else if (name == "reset") {
    if (py::isinstance<py::list>(obj.attr("pos"))) {
      positions = obj.attr("pos").cast<vector<pos_t>>();
    } else if (py::isinstance<py::int_>(obj.attr("pos"))) {
      positions = vector<pos_t>{obj.attr("pos").cast<pos_t>()};
    }
    return QuantumOperator(name, positions);

  } else if (name == "cif") {
    uint condition = 0;
    vector<QuantumOperator> instructions;
    if (py::isinstance<py::list>(obj.attr("cbits"))) {
      cbits = obj.attr("cbits").cast<vector<pos_t>>();
    } else if (py::isinstance<py::int_>(obj.attr("cbits"))) {
      cbits = vector<pos_t>{obj.attr("cbits").cast<pos_t>()};
    }

    if (py::isinstance<py::int_>(obj.attr("condition"))) {
      condition = obj.attr("condition").cast<pos_t>();
    }

    // Recursively handdle instruction
    if (py::isinstance<py::list>(obj.attr("instructions"))) {
      auto pyops = obj.attr("instructions");
      for (auto pyop_h : pyops) {
        py::object pyop = py::reinterpret_borrow<py::object>(pyop_h);
        QuantumOperator op = from_pyops(pyop);
        if (op) {
          if (op.targe_num() > max_targe_num_)
            max_targe_num_ = op.targe_num();
          instructions.push_back(std::move(op));
        }
      }
    }
    return QuantumOperator(name, cbits, condition, instructions);

  } else {
    return QuantumOperator();
  }
}

Circuit::Circuit(py::object const& pycircuit) : max_targe_num_(0) {
  // auto pygates = pycircuit.attr("gates");
  auto pyops = pycircuit.attr("instructions");
  auto used_qubits = pycircuit.attr("used_qubits").cast<vector<pos_t>>();
  cbit_num_ = pycircuit.attr("cbits_num").cast<uint>();
  qubit_num_ = *std::max_element(used_qubits.begin(), used_qubits.end()) + 1;
  // judge wheather op qubit after measure
  bool measured = false;
  for (auto pyop_h : pyops) {
    py::object pyop = py::reinterpret_borrow<py::object>(pyop_h);
    QuantumOperator op = from_pyops(pyop);
    if (op) {
      if (op.targe_num() > max_targe_num_)
        max_targe_num_ = op.targe_num();
      if (op.name() == "measure") {
        measured = true;
      } else if (measured == true) {
        final_measure_ = false;
      }
      instructions_.push_back(std::move(op));
    }
  }
}

void Circuit::compress_instructions() {}

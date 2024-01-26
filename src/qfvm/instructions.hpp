#pragma once

#include "statevector.hpp"
#include "qasm.hpp"
#include <iostream>

class Instruction {
    protected:
    string name_ = "empty";
    vector<pos_t> positions_;
  
  public:
  string name() const { return name_; }
  vector<pos_t> positions() { return positions_; }
  explicit operator bool() const {
        return !(name_ == "empty");
    }
};


class QuantumOperator :  public Instruction {
    protected:
        vector<double> paras_;
        uint control_num_;
        uint targe_num_;
        bool diag_;
        bool real_;
        RowMatrixXcd mat_;

    //Constructor
    QuantumOperator();
    QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat = RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0), bool diag=false, bool real=false);

    QuantumOperator(string const name,vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat=RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0), bool diag=false, bool real=false);

    QuantumOperator(string const name, vector<pos_t> const &positions, RowMatrixXcd const &mat=RowMatrixXcd(0, 0), RowMatrixXcd const &full_mat = RowMatrixXcd(0, 0));
    // virtual ~QuantumOperator();

    //data accessor
    vector<double> paras() const {return paras_;}
    bool has_control() const{return control_num_ == 0 ? false : true;}
    bool is_real() const{ return real_; }
    bool is_diag() const{ return diag_; }
    RowMatrixXcd targ_mat() const { return targ_mat_;}
    RowMatrixXcd full_mat() const {return full_mat_;}
    uint control_num() const { return control_num_; } 
    uint targe_num() const { return targe_num_; }
}

QuantumOperator::QuantumOperator() : name_("empty"){ };

QuantumOperator::QuantumOperator(string const name, vector<pos_t> const &positions, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat)
:
name_(name),
paras_(vector<double>(0)),
positions_(positions),
control_num_(-1),
targe_num_(0),
diag_(false),
real_(false),
targ_mat_(mat),
full_mat_(full_mat){ }

QuantumOperator::QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &positions, uint control_num, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat, bool diag, bool real)
:
name_(name),
paras_(paras),
positions_(positions),
control_num_(control_num),
targe_num_(positions.size()-control_num),
diag_(diag),
real_(real),
targ_mat_(mat), 
full_mat_(full_mat){ }

QuantumOperator::QuantumOperator(string const name, vector<double> paras, vector<pos_t> const &control_qubits, vector<pos_t> const &targe_qubits, RowMatrixXcd const &mat, RowMatrixXcd const &full_mat, bool diag, bool real)
:
name_(name),
paras_(paras),
diag_(diag),
real_(real),
targ_mat_(mat),
full_mat_(full_mat){
    positions_ = control_qubits;
    positions_.insert(positions_.end(), targe_qubits.begin(), targe_qubits.end());
    control_num_ = control_qubits.size();
    targe_num_ = targe_qubits.size();
}

class Measures : public Instruction {
    private:
      static const string name_ = "measure";
    protected:
      vector<pos_t> qbits_;
      vector<pos_t> cbits_;
      
    pubic:
      MeasuresOp(){ };
      vector<pos_t> qbits() { return qbits_; }
      vector<pos_t> cbits() { return cbits_; }
};


class Reset : public Instruction{
  private:
      static const string name_ = "reset";
}

class Cif : public Instruction{
  private:
      static const string name_ = "cif";
  protected:
    vector<pos_t> cbits_;
    uint condition_;
    vector<QuantumOperator> instructions;

    uint condition() const { return condition_; }
    vector<QuantumOperator> instructions() { return instructions_; }
    vector<pos_t> cbits() { return cbits_; }
}

// class QuantumOperator {
// protected:
//   string name_;
//   vector<pos_t> positions_;
//   vector<double> paras_;
//   uint control_num_;
//   uint targe_num_;
//   bool diag_;
//   bool real_;
//   RowMatrixXcd mat_;
//   vector<pos_t> qbits_;
//   vector<pos_t> cbits_;
//   vector<QuantumOperator> instructions_;
//   uint condition_;

// public:
//   // Constructor
//   QuantumOperator();
//   QuantumOperator(string name, vector<pos_t> const& qbits);
//   QuantumOperator(string name, vector<pos_t> const& qbits,
//                   vector<pos_t> const& cbits);
//   QuantumOperator(string name, vector<pos_t> const& cbits, const uint condition,
//                   vector<QuantumOperator> const& ins);
//   QuantumOperator(string name, vector<double> paras,
//                   vector<pos_t> const& control_qubits,
//                   vector<pos_t> const& targe_qubits, RowMatrixXcd const& mat,
//                   bool diag = false, bool real = false);
//   QuantumOperator(string name, vector<double> paras,
//                   vector<pos_t> const& positions, uint control_num,
//                   RowMatrixXcd const& mat, bool diag = false,
//                   bool real = false);

//   // data accessor
//   string name() const { return name_; }
//   vector<double> paras() const { return paras_; }
//   bool has_control() const { return control_num_ == 0 ? false : true; }
//   bool is_real() const { return real_; }
//   bool is_diag() const { return diag_; }
//   RowMatrixXcd mat() const { return mat_; }
//   uint control_num() const { return control_num_; }
//   uint targe_num() const { return targe_num_; }
//   uint condition() const { return condition_; }
//   vector<pos_t> positions() { return positions_; }
//   explicit operator bool() const { return !(name_ == "empty"); }
//   vector<pos_t> qbits() { return qbits_; }
//   vector<pos_t> cbits() { return cbits_; }
//   vector<QuantumOperator> instructions() { return instructions_; }
//   // Apply method
//   virtual void apply_to_state(StateVector<double>& state){};
// };

// QuantumOperator::QuantumOperator() : name_("empty"){};

// QuantumOperator::QuantumOperator(string name, vector<pos_t> const& qbits)
//     : name_(name), targe_num_(0), qbits_(qbits) {}

// QuantumOperator::QuantumOperator(string name, vector<pos_t> const& qbits,
//                                  vector<pos_t> const& cbits)
//     : name_(name), targe_num_(0), qbits_(qbits), cbits_(cbits) {}

// QuantumOperator::QuantumOperator(string name, vector<pos_t> const& cbits,
//                                  const uint condition,
//                                  vector<QuantumOperator> const& ins)
//     : name_(name), targe_num_(0), cbits_(cbits), instructions_(ins),
//       condition_(condition) {}

// QuantumOperator::QuantumOperator(string name, vector<double> paras,
//                                  vector<pos_t> const& positions,
//                                  uint control_num, RowMatrixXcd const& mat,
//                                  bool diag, bool real)
//     : name_(name), paras_(paras), positions_(positions),
//       control_num_(control_num), targe_num_(positions.size() - control_num),
//       diag_(diag), real_(real), mat_(mat) {}

// QuantumOperator::QuantumOperator(string name, vector<double> paras,
//                                  vector<pos_t> const& control_qubits,
//                                  vector<pos_t> const& targe_qubits,
//                                  RowMatrixXcd const& mat, bool diag, bool real)
//     : name_(name), paras_(paras), diag_(diag), real_(real), mat_(mat) {
//   positions_ = control_qubits;
//   positions_.insert(positions_.end(), targe_qubits.begin(), targe_qubits.end());
//   control_num_ = control_qubits.size();
//   targe_num_ = targe_qubits.size();
// }


// Construct C++ operators from pygates
std::unique_ptr<Instruction> from_pyops(py::object const& obj) {
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
        vector<Instruction> instructions;
        cbits = obj.attr("cbits").cast<vector<pos_t>>();
        condition = obj.attr("condition").cast<pos_t>();

        // Recursively handdle instruction
        if (py::isinstance<py::list>(obj.attr("instructions"))) {
            auto pyops = obj.attr("instructions");
            for (auto pyop_h : pyops) {
                py::object pyop = py::reinterpret_borrow<py::object>(pyop_h);
                std::unique_ptr<Instruction> ins = from_pyops(pyop);
                if (*ins) 
                    instructions.push_back(std::move(*ins));
                }
        }
        return  std::make_unique<Cif>(name, cbits, condition, instructions);

    } else {
        return  std::make_unique<QuantumOperator>();
    }
}

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
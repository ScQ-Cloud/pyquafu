#pragma once

#include "operators.hpp"
#include "qasm.hpp"

class Circuit{
    private:
        uint qubit_num_;    
        vector<QuantumOperator> gates_{0};

    public:
    Circuit();
    Circuit(uint qubit_num);
    Circuit(vector<QuantumOperator> gates);

    void add_gate(QuantumOperator gate);
    void build_from_qasm(string qasm);
    void compress_gates();

    const uint qubit_num(){ return qubit_num_; }
    const vector<QuantumOperator>gates(){ return gates_; }

};

void Circuit::add_gate(QuantumOperator gate){
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

 Circuit::Circuit(vector<QuantumOperator> gates)
 :
 gates_(gates){
    qubit_num_ = 0;
    for (auto gate : gates){
        for (pos_t pos : gate.positions()){
            if (pos > qubit_num_){ qubit_num_ = pos; }
        }
    }
}

void Circuit::build_from_qasm(string qasm){
    auto lines = split_string(qasm, '\n');
    for (auto i=2;i<lines.size();i++){
        auto op = compile_line(lines[i]);
        switch (OPMAP[op.name]){
            case Opname::x:
                add_gate(QuantumOperator(op.name, op.positions, 0, {0, 1, 1, 0}));
                break;
            case Opname::y:
                break;
        }
    }
}

void Circuit::compress_gates(){}

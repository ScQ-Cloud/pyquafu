#pragma once

#include <unordered_map>
#include "types.hpp"
#include "util.h"

using Qfutil::split_string;
using Qfutil::find_numbers;
#define Pair(name) {#name, Opname::name}

enum class Opname{
    creg, x, y, z, h, s, sdg, t, tdg, p, rx, ry, rz, cnot, cx, cz, crx, cp, ccx, toffoli, swap, iswap, rxx, ryy, rzz, measure, reset, cif
};

std::unordered_map<string, Opname> OPMAP{Pair(creg), Pair(x), Pair(y), Pair(z), Pair(h), Pair(s), Pair(sdg), Pair(t),
                            Pair(tdg), Pair(p), Pair(rx), Pair(ry), Pair(rz), Pair(cnot), Pair(cx), Pair(cz), 
                            Pair(crx), Pair(cp), Pair(ccx), Pair(swap), Pair(iswap), Pair(rxx), Pair(ryy), 
                            Pair(rzz), Pair(measure), Pair(reset), Pair(cif)};

typedef struct{
    string name;
    vector<pos_t> positions;
    vector<double> params; 

    void print_info(){
        std::cout << "name " << name << std::endl;
        std::cout << "positions: ";
        for (auto pos : positions){
            std::cout << pos << " ";
        }
        std::cout << std::endl;

        if (params.size() > 0){
        printf("parameters: ");
            for (auto para : params){
                printf("%.6f ", para);
            }
        }
        printf("\n");
        printf("-----\n");
    }

} Operation;

Operation compile_line(string const& line){
    auto operation_qbits = split_string(line, ' ', 1);
    auto operation = operation_qbits[0];
    auto qbits = operation_qbits[1];
    auto positions = find_numbers<pos_t>(qbits);
    auto opname_params = split_string(operation, '(', 1);
    auto opname = opname_params[0]; 
    vector<double> params;
    if (opname_params.size() > 1){
        params = find_numbers<double>(opname_params[1]);
    }

    return Operation{opname, positions, params};
}
#pragma once

#include "types.hpp"
#include "util.h"
#include <unordered_map>

using Qfutil::find_numbers;
using Qfutil::split_string;
#define Pair(name)                                                             \
  {                                                                            \
#name, Opname::name                                                        \
  }


/****This is C++-statevector-native gate set, it has not to be consistent with pyquafu. It is used to avoid matrix copy from py at present. Of course providing more native gate is needed in the future, e.g. efficient swap-like gate.****/
enum class Opname {
  creg,
  x,
  y,
  z,
  h,
  s,
  sdg,
  t,
  tdg,
  p,
  rx,
  ry,
  rz,
  cnot,
  cx,
  cz,
  cp,
  ccx,
  toffoli,
  rzz,
  measure,
  reset,
  cif
};

std::unordered_map<string, Opname> OPMAP{
    Pair(creg),    Pair(x),     Pair(y),     Pair(z),   Pair(h),   Pair(s),
    Pair(sdg),     Pair(t),     Pair(tdg),   Pair(p),   Pair(rx),  Pair(ry),
    Pair(rz),      Pair(cnot),  Pair(cx),    Pair(cz),  Pair(cp),
    Pair(ccx),     Pair(rzz),
    Pair(measure), Pair(reset), Pair(cif)};

struct Operation {
  string name;
  vector<pos_t> positions;
  vector<double> params;

  void print_info() {
    std::cout << "name " << name << std::endl;
    std::cout << "positions: ";
    for (auto pos : positions) {
      std::cout << pos << " ";
    }
    std::cout << std::endl;

    if (params.size() > 0) {
      printf("parameters: ");
      for (auto para : params) {
        printf("%.6f ", para);
      }
    }
    printf("\n");
    printf("-----\n");
  }
};

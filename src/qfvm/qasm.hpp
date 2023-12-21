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
  crx,
  cp,
  ccx,
  toffoli,
  swap,
  iswap,
  rxx,
  ryy,
  rzz,
  measure,
  reset,
  cif
};

std::unordered_map<string, Opname> OPMAP{
    Pair(creg),    Pair(x),     Pair(y),     Pair(z),   Pair(h),   Pair(s),
    Pair(sdg),     Pair(t),     Pair(tdg),   Pair(p),   Pair(rx),  Pair(ry),
    Pair(rz),      Pair(cnot),  Pair(cx),    Pair(cz),  Pair(crx), Pair(cp),
    Pair(ccx),     Pair(swap),  Pair(iswap), Pair(rxx), Pair(ryy), Pair(rzz),
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

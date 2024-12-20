#pragma once

#include <time.h>

#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <regex>
#include <set>
#include <type_traits>
#include <vector>

#include "types.hpp"

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                                                                        \
    std::cout << #x ": "                                                                                               \
              << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()            \
                                                                           - bench_##x)                                \
                     .count()                                                                                          \
              << "s" << std::endl;

namespace Qfutil {

std::vector<int> randomArr(size_t length, size_t max) {
    srand((unsigned) time(NULL));
    std::vector<int> arr(length);
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = rand() % max;
    }
    return arr;
}

std::vector<double> randomDoubleArr(size_t length) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0., 1.);

    std::vector<double> randarr;
    for (auto n = 0; n < length; ++n) {
        randarr.push_back(distr(eng));
    }
    return randarr;
}

int randomint(int min, int max) {
    srand((unsigned) time(NULL));
    return (rand() % (max - min + 1)) + min;
}

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4eb2d;
    i ^= i >> 15;
    return i;
}

template <class T>
void printVector(std::vector<T> const& arr) {
    for (auto i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

std::vector<std::string> split_string(const std::string& str, char delim) {
    std::vector<std::string> elems;
    auto lastPos = str.find_first_not_of(delim, 0);
    auto pos = str.find_first_of(delim, lastPos);
    while (pos != std::string::npos || lastPos != std::string::npos) {
        elems.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delim, pos);
        pos = str.find_first_of(delim, lastPos);
    }
    return elems;
}

std::vector<std::string> split_string(const std::string& str, char delim, uint num) {
    auto end = str.length();
    std::vector<std::string> elems;
    auto lastPos = str.find_first_not_of(delim, 0);
    auto pos = str.find_first_of(delim, lastPos);
    while ((pos != std::string::npos || lastPos != std::string::npos) && elems.size() < num) {
        elems.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delim, pos);
        pos = str.find_first_of(delim, lastPos);
    }

    if ((pos != std::string::npos || lastPos != std::string::npos)) {
        elems.push_back(str.substr(lastPos, end));
    }
    return elems;
}

template <class real_t>
std::vector<real_t> find_numbers(const std::string& str) {
    std::smatch matchs;
    std::vector<real_t> res;
    std::regex pattern;
    if (std::is_unsigned<real_t>::value) {
        pattern = std::regex("\\d+");
    } else if (std::is_floating_point<real_t>::value) {
        pattern = std::regex("-?(([1-9]\\d*\\.\\d*)|(0\\.\\d*[1-9]\\d*))|\\d+");
    }

    auto begin = std::sregex_iterator(str.begin(), str.end(), pattern);
    const std::sregex_iterator end;

    for (std::sregex_iterator i = begin; i != end; ++i) {
        std::string match_str = i->str();
        if (std::is_unsigned<real_t>::value) {
            res.push_back(std::stoi(match_str));
        } else if (std::is_floating_point<real_t>::value) {
            res.push_back(std::stod(match_str));
        }
    }

    return res;
}

/*----------------bit function------------------*/
const std::complex<double> PHASE_YZ[4] = {1, imag_I, -1, -imag_I};

inline static uint popcount(uint x) {
    x = ((x & 0xaaaaaaaaaaaaaaaaUL) >> 1) + (x & 0x5555555555555555UL);
    x = ((x & 0xccccccccccccccccUL) >> 2) + (x & 0x3333333333333333UL);
    x = ((x & 0xf0f0f0f0f0f0f0f0UL) >> 4) + (x & 0x0f0f0f0f0f0f0f0fUL);
    x = ((x & 0xff00ff00ff00ff00UL) >> 8) + (x & 0x00ff00ff00ff00ffUL);
    x = ((x & 0xffff0000ffff0000UL) >> 16) + (x & 0x0000ffff0000ffffUL);
    x = ((x & 0xffffffff00000000UL) >> 32) + (x & 0x00000000ffffffffUL);
    return (uint) x;
}

}  // namespace Qfutil

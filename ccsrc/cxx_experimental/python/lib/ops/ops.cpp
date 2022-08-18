//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Qubit.h>

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops/gates.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/qubit_operator_parameter_resolver.hpp"
#include "ops/parametric/angle_gates.hpp"
#include "python/bindings.hpp"
#include "python/ops/gate_adapter.hpp"

namespace ops = mindquantum::ops;
namespace py = pybind11;

namespace {
// NB: These two should have auto return types but GCC 7 & 8 don't play well if we do :-(
template <typename operator_t>
std::string_view to_string(const operator_t& op) {
    return operator_t::kind();
}
template <typename operator_t>
std::string to_string_angle(const operator_t& op) {
    return fmt::format("{}", operator_t::kind(), op.angle());
}
}  // namespace

void init_tweedledum_ops(pybind11::module& module) {
    py::class_<ops::Barrier>(module, "Barrier").def(py::init<>()).def("__str__", &::to_string<ops::Barrier>);
    py::class_<ops::H>(module, "H").def(py::init<>()).def("__str__", &::to_string<ops::H>);
    py::class_<ops::Measure>(module, "Measure").def(py::init<>()).def("__str__", &::to_string<ops::Measure>);
    py::class_<ops::S>(module, "S").def(py::init<>()).def("__str__", &::to_string<ops::S>);
    py::class_<ops::Sdg>(module, "Sdg").def(py::init<>()).def("__str__", &::to_string<ops::Sdg>);
    py::class_<ops::Swap>(module, "Swap").def(py::init<>()).def("__str__", &::to_string<ops::Swap>);
    py::class_<ops::Sx>(module, "Sx").def(py::init<>()).def("__str__", &::to_string<ops::Sx>);
    py::class_<ops::Sxdg>(module, "Sxdg").def(py::init<>()).def("__str__", &::to_string<ops::Sxdg>);
    py::class_<ops::T>(module, "T").def(py::init<>()).def("__str__", &::to_string<ops::T>);
    py::class_<ops::Tdg>(module, "Tdg").def(py::init<>()).def("__str__", &::to_string<ops::Tdg>);
    py::class_<ops::X>(module, "X").def(py::init<>()).def("__str__", &::to_string<ops::X>);
    py::class_<ops::Y>(module, "Y").def(py::init<>()).def("__str__", &::to_string<ops::Y>);
    py::class_<ops::Z>(module, "Z").def(py::init<>()).def("__str__", &::to_string<ops::Z>);

    py::class_<ops::P>(module, "P").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::P>);
    py::class_<ops::Rx>(module, "Rx").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rx>);
    py::class_<ops::Rxx>(module, "Rxx").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rxx>);
    py::class_<ops::Ry>(module, "Ry").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Ry>);
    py::class_<ops::Ryy>(module, "Ryy").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Ryy>);
    py::class_<ops::Rz>(module, "Rz").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rz>);
    py::class_<ops::Rzz>(module, "Rzz").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rzz>);
}

void init_mindquantum_ops(pybind11::module& module) {
    using namespace pybind11::literals;

    py::class_<ops::SqrtSwap>(module, "SqrtSwap").def(py::init<>()).def("__str__", &::to_string<ops::SqrtSwap>);

    py::class_<ops::Entangle>(module, "Entangle")
        .def(py::init<const uint32_t>())
        .def("__str__", &::to_string<ops::Entangle>);
    py::class_<ops::Ph>(module, "Ph").def(py::init<const double>());
    py::class_<ops::QFT>(module, "QFT").def(py::init<const uint32_t>()).def("__str__", &::to_string<ops::QFT>);

    py::enum_<ops::TermValue>(module, "TermValue")
        .value("I", ops::TermValue::I)
        .value("X", ops::TermValue::X)
        .value("Y", ops::TermValue::Y)
        .value("Z", ops::TermValue::Z)
        .value("a", ops::TermValue::a)
        .value("adg", ops::TermValue::adg);

    // =========================================================================

#define TO_STRING1(X)    #X
#define TO_STRING(X)     TO_STRING1(X)
#define CONCAT2(A, B)    A##B
#define CONCAT3(A, B, C) A##B##C

#define PYBIND11_DEFINE_BINOP_IMPL(py_name, lhs_t, rhs_t, op)                                                          \
    def(                                                                                                               \
        py_name, [](lhs_t& lhs, rhs_t rhs) { return lhs op rhs; }, py::is_operator())
#define PYBIND11_DEFINE_BINOP(py_name, lhs_t, rhs_t, op) PYBIND11_DEFINE_BINOP_IMPL(#py_name, lhs_t, rhs_t, op)
#define PYBIND11_DEFINE_BINOP_PAIR(py_name, lhs_t, rhs_t, op)                                                          \
    PYBIND11_DEFINE_BINOP_IMPL(TO_STRING(CONCAT3(__i, py_name, __)), lhs_t, rhs_t, CONCAT2(op, =))                     \
        .PYBIND11_DEFINE_BINOP_IMPL(TO_STRING(CONCAT3(__, py_name, __)), const lhs_t, rhs_t, op)
#define PYBIND11_DEFINE_UNOP(py_name, lhs_t, op)                                                                       \
    def(                                                                                                               \
        #py_name, [](const lhs_t& base) { return op base; }, py::is_operator())

    py::class_<ops::QubitOperator>(module, "QubitOperator")
        .def(py::init<>())
        .def(py::init<const ops::term_t&, ops::QubitOperator::coefficient_t>(), "term"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::terms_t&, ops::QubitOperator::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::py_terms_t&, ops::QubitOperator::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::QubitOperator::coeff_term_dict_t&>(), "coeff_terms"_a)
        .def(py::init<std::string_view, ops::QubitOperator::coefficient_t>(), "terms_string"_a, "coeff"_a = 1.0)
        .def("num_targets", &ops::QubitOperator::num_targets)
        .def("count_qubits", &ops::QubitOperator::count_qubits)
        .def("is_identity", &ops::QubitOperator::is_identity, "abs_tol"_a = ops::QubitOperator::EQ_TOLERANCE)
        .def_static("identity", &ops::QubitOperator::identity)
        .def("constant", static_cast<void (ops::QubitOperator::*)(const ops::QubitOperator::coefficient_t&)>(
                             &ops::QubitOperator::constant))
        .def("constant", static_cast<ops::QubitOperator::coefficient_t (ops::QubitOperator::*)() const>(
                             &ops::QubitOperator::constant))
        .def("is_singlet", &ops::QubitOperator::real)
        .def("singlet", &ops::QubitOperator::real)
        .def("singlet_coeff", &ops::QubitOperator::real)
        .def("split", &ops::QubitOperator::real)
        .def("imag", &ops::QubitOperator::imag)
        .def("compress", &ops::QubitOperator::compress, "abs_tol"_a = ops::QubitOperator::EQ_TOLERANCE)
        .def("dumps", &ops::QubitOperator::dumps, "indent"_a = 4)
        .def_static("loads", ops::QubitOperator::loads, "string_data"_a)
        .def(
            "__str__", [](const ops::QubitOperator& base) { return base.to_string(); }, py::is_operator())
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperator, const ops::QubitOperator&, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperator, double, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperator, std::complex<double>, +)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperator, const ops::QubitOperator&, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperator, double, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperator, std::complex<double>, -)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperator, const ops::QubitOperator&, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperator, double, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperator, std::complex<double>, *)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::QubitOperator, double, /)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::QubitOperator, std::complex<double>, /)
        .PYBIND11_DEFINE_UNOP(__neg__, ops::QubitOperator, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const ops::QubitOperator, const ops::QubitOperator&, ==)
        .def(
            "__pow__", [](const ops::QubitOperator& base, unsigned int exponent) { return base.pow(exponent); },
            py::is_operator())
        .def("count_gates", &ops::QubitOperator::count_gates)
        .def("matrix", &ops::QubitOperator::matrix, "n_qubits"_a);

    py::class_<ops::QubitOperatorPR>(module, "QubitOperatorPR")
        .def(py::init<>())
        .def(py::init<const ops::term_t&, ops::QubitOperatorPR::coefficient_t>(), "term"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::terms_t&, ops::QubitOperatorPR::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::py_terms_t&, ops::QubitOperatorPR::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::QubitOperatorPR::coeff_term_dict_t&>(), "coeff_terms"_a)
        .def(py::init<std::string_view, ops::QubitOperatorPR::coefficient_t>(), "terms_string"_a, "coeff"_a = 1.0)
        .def("num_targets", &ops::QubitOperatorPR::num_targets)
        .def("count_qubits", &ops::QubitOperatorPR::count_qubits)
        .def("is_identity", &ops::QubitOperatorPR::is_identity, "abs_tol"_a = ops::QubitOperatorPR::EQ_TOLERANCE)
        .def_static("identity", &ops::QubitOperatorPR::identity)
        .def("constant", static_cast<void (ops::QubitOperatorPR::*)(const ops::QubitOperatorPR::coefficient_t&)>(
                             &ops::QubitOperatorPR::constant))
        .def("constant", static_cast<ops::QubitOperatorPR::coefficient_t (ops::QubitOperatorPR::*)() const>(
                             &ops::QubitOperatorPR::constant))
        .def("is_singlet", &ops::QubitOperatorPR::real)
        .def("singlet", &ops::QubitOperatorPR::real)
        .def("singlet_coeff", &ops::QubitOperatorPR::real)
        .def("split", &ops::QubitOperatorPR::real)
        .def("imag", &ops::QubitOperatorPR::imag)
        .def("compress", &ops::QubitOperatorPR::compress, "abs_tol"_a = ops::QubitOperatorPR::EQ_TOLERANCE)
        .def("dumps", &ops::QubitOperatorPR::dumps, "indent"_a = 4)
        .def_static("loads", ops::QubitOperatorPR::loads, "string_data"_a)
        .def(
            "__str__", [](const ops::QubitOperatorPR& base) { return base.to_string(); }, py::is_operator())
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperatorPR, const ops::QubitOperatorPR&, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperatorPR, double, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::QubitOperatorPR, std::complex<double>, +)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperatorPR, const ops::QubitOperatorPR&, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperatorPR, double, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::QubitOperatorPR, std::complex<double>, -)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperatorPR, const ops::QubitOperatorPR&, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperatorPR, double, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::QubitOperatorPR, std::complex<double>, *)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::QubitOperatorPR, double, /)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::QubitOperatorPR, std::complex<double>, /)
        .PYBIND11_DEFINE_UNOP(__neg__, ops::QubitOperatorPR, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const ops::QubitOperatorPR, const ops::QubitOperatorPR&, ==)
        .def(
            "__pow__", [](const ops::QubitOperatorPR& base, unsigned int exponent) { return base.pow(exponent); },
            py::is_operator())
        .def("count_gates", &ops::QubitOperatorPR::count_gates)
        .def("matrix", &ops::QubitOperatorPR::matrix, "n_qubits"_a);

    py::class_<ops::FermionOperator>(module, "FermionOperator")
        .def(py::init<>())
        .def(py::init<const ops::term_t&, ops::FermionOperator::coefficient_t>(), "term"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::terms_t&, ops::FermionOperator::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::py_terms_t&, ops::FermionOperator::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::FermionOperator::coeff_term_dict_t&>(), "coeff_terms"_a)
        .def(py::init<std::string_view, ops::FermionOperator::coefficient_t>(), "terms_string"_a, "coeff"_a = 1.0)
        .def("num_targets", &ops::FermionOperator::num_targets)
        .def("is_identity", &ops::FermionOperator::is_identity, "abs_tol"_a = ops::FermionOperator::EQ_TOLERANCE)
        .def_static("identity", &ops::FermionOperator::identity)
        .def("constant", static_cast<void (ops::FermionOperator::*)(const ops::FermionOperator::coefficient_t&)>(
                             &ops::FermionOperator::constant))
        .def("constant", static_cast<ops::FermionOperator::coefficient_t (ops::FermionOperator::*)() const>(
                             &ops::FermionOperator::constant))
        .def("is_singlet", &ops::FermionOperator::real)
        .def("singlet", &ops::FermionOperator::real)
        .def("singlet_coeff", &ops::FermionOperator::real)
        .def("split", &ops::FermionOperator::real)
        .def("imag", &ops::FermionOperator::imag)
        .def("compress", &ops::FermionOperator::compress, "abs_tol"_a = ops::FermionOperator::EQ_TOLERANCE)
        .def("dumps", &ops::FermionOperator::dumps, "indent"_a = 4)
        .def_static("loads", ops::FermionOperator::loads, "string_data"_a)
        .def(
            "__str__", [](const ops::FermionOperator& base) { return base.to_string(); }, py::is_operator())
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperator, const ops::FermionOperator&, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperator, double, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperator, std::complex<double>, +)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperator, const ops::FermionOperator&, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperator, double, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperator, std::complex<double>, -)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperator, const ops::FermionOperator&, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperator, double, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperator, std::complex<double>, *)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::FermionOperator, double, /)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::FermionOperator, std::complex<double>, /)
        .PYBIND11_DEFINE_UNOP(__neg__, ops::FermionOperator, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const ops::FermionOperator, const ops::FermionOperator&, ==)
        .def(
            "__pow__", [](const ops::FermionOperator& base, unsigned int exponent) { return base.pow(exponent); },
            py::is_operator())
        .def("matrix", &ops::FermionOperator::matrix, "n_qubits"_a)
        .def("normal_ordered", &ops::FermionOperator::normal_ordered);

    py::class_<ops::FermionOperatorPR>(module, "FermionOperatorPR")
        .def(py::init<>())
        .def(py::init<const ops::term_t&, ops::FermionOperatorPR::coefficient_t>(), "term"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::terms_t&, ops::FermionOperatorPR::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::py_terms_t&, ops::FermionOperatorPR::coefficient_t>(), "terms"_a, "coeff"_a = 1.0)
        .def(py::init<const ops::FermionOperatorPR::coeff_term_dict_t&>(), "coeff_terms"_a)
        .def(py::init<std::string_view, ops::FermionOperatorPR::coefficient_t>(), "terms_string"_a, "coeff"_a = 1.0)
        .def("num_targets", &ops::FermionOperatorPR::num_targets)
        .def("is_identity", &ops::FermionOperatorPR::is_identity, "abs_tol"_a = ops::FermionOperatorPR::EQ_TOLERANCE)
        .def_static("identity", &ops::FermionOperatorPR::identity)
        .def("constant", static_cast<void (ops::FermionOperatorPR::*)(const ops::FermionOperatorPR::coefficient_t&)>(
                             &ops::FermionOperatorPR::constant))
        .def("constant", static_cast<ops::FermionOperatorPR::coefficient_t (ops::FermionOperatorPR::*)() const>(
                             &ops::FermionOperatorPR::constant))
        .def("is_singlet", &ops::FermionOperatorPR::real)
        .def("singlet", &ops::FermionOperatorPR::real)
        .def("singlet_coeff", &ops::FermionOperatorPR::real)
        .def("split", &ops::FermionOperatorPR::real)
        .def("imag", &ops::FermionOperatorPR::imag)
        .def("compress", &ops::FermionOperatorPR::compress, "abs_tol"_a = ops::FermionOperatorPR::EQ_TOLERANCE)
        .def("dumps", &ops::FermionOperatorPR::dumps, "indent"_a = 4)
        .def_static("loads", ops::FermionOperatorPR::loads, "string_data"_a)
        .def(
            "__str__", [](const ops::FermionOperatorPR& base) { return base.to_string(); }, py::is_operator())
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperatorPR, const ops::FermionOperatorPR&, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperatorPR, double, +)
        .PYBIND11_DEFINE_BINOP_PAIR(add, ops::FermionOperatorPR, std::complex<double>, +)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperatorPR, const ops::FermionOperatorPR&, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperatorPR, double, -)
        .PYBIND11_DEFINE_BINOP_PAIR(sub, ops::FermionOperatorPR, std::complex<double>, -)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperatorPR, const ops::FermionOperatorPR&, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperatorPR, double, *)
        .PYBIND11_DEFINE_BINOP_PAIR(mul, ops::FermionOperatorPR, std::complex<double>, *)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::FermionOperatorPR, double, /)
        .PYBIND11_DEFINE_BINOP_PAIR(truediv, ops::FermionOperatorPR, std::complex<double>, /)
        .PYBIND11_DEFINE_UNOP(__neg__, ops::FermionOperatorPR, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const ops::FermionOperatorPR, const ops::FermionOperatorPR&, ==)
        .def(
            "__pow__", [](const ops::FermionOperatorPR& base, unsigned int exponent) { return base.pow(exponent); },
            py::is_operator())
        .def("matrix", &ops::FermionOperatorPR::matrix, "n_qubits"_a)
        .def("normal_ordered", &ops::FermionOperatorPR::normal_ordered);

#undef TO_STRING1
#undef TO_STRING
#undef CONCAT2
#undef CONCAT3
#undef PYBIND11_DEFINE_BINOP
#undef PYBIND11_DEFINE_BINOP_IMPL
#undef PYBIND11_DEFINE_BINOP_PAIR
#undef PYBIND11_DEFINE_UNOP

    // =========================================================================

    // py::class_<ops::parametric::P>(module, "P").def(py::init<const double>());
    // py::class_<ops::parametric::Ph>(module, "Ph"). def(py::init<SymEngine::number>());
    // py::class_<ops::parametric::Rx>(module, "Rx").def(py::init<const double>());
    // py::class_<ops::parametric::Rxx>(module, "Rxx").def(py::init<const double>());
    // py::class_<ops::parametric::Ry>(module, "Ry").def(py::init<const double>());
    // py::class_<ops::parametric::Ryy>(module, "Ryy").def(py::init<const double>());
    // py::class_<ops::parametric::Rz>(module, "Rz").def(py::init<const double>());
    // py::class_<ops::parametric::Rzz>(module, "Rzz").def(py::init<const double>());
}

void mindquantum::python::init_ops(pybind11::module& module) {
    init_tweedledum_ops(module);
    init_mindquantum_ops(module);
}

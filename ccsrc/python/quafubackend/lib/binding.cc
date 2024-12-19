/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "python/device/binding.h"

#include <complex>
#include <memory>

#include <fmt/format.h>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config/constexpr_type_name.h"
#include "config/format/std_complex.h"
#include "config/type_traits.h"
#include "core/quafu_base_types.h"
#include "core/sparse/algo.h"
#include "core/sparse/csrhdmatrix.h"
#include "core/sparse/paulimat.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"

#include "python/core/sparse/csrhdmatrix.h"
#include "python/ops/basic_gate.h"
#include "python/ops/build_env.h"

namespace py = pybind11;

using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

namespace quafu::python {
void init_logging(pybind11::module &module);  // NOLINT(runtime/references)NOLINT

void BindTypeIndependentGate(py::module &module) {  // NOLINT(runtime/references)
    using quafu::Index;
    using quafu::qbits_t;
    using quafu::VT;
    py::class_<quafu::MeasureGate, quafu::BasicGate, std::shared_ptr<quafu::MeasureGate>>(module, "MeasureGate")
        .def(py::init<std::string, const qbits_t &>(), "name"_a, "obj_qubits"_a)
        .def(py::init<std::string, const qbits_t &, quafu::index_t>(), "name"_a, "obj_qubits"_a, "reset_to"_a);
    py::class_<quafu::IGate, quafu::BasicGate, std::shared_ptr<quafu::IGate>>(module, "IGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::XGate, quafu::BasicGate, std::shared_ptr<quafu::XGate>>(module, "XGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::YGate, quafu::BasicGate, std::shared_ptr<quafu::YGate>>(module, "YGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::ZGate, quafu::BasicGate, std::shared_ptr<quafu::ZGate>>(module, "ZGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::PauliString, quafu::BasicGate, std::shared_ptr<quafu::PauliString>>(module, "GroupedPauli")
        .def(py::init<const std::string &, const qbits_t &, const qbits_t &>(), "paulis"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::HGate, quafu::BasicGate, std::shared_ptr<quafu::HGate>>(module, "HGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::ISWAPGate, quafu::BasicGate, std::shared_ptr<quafu::ISWAPGate>>(module, "ISWAPGate")
        .def(py::init<bool, const qbits_t &, const qbits_t &>(), "daggered"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SWAPGate, quafu::BasicGate, std::shared_ptr<quafu::SWAPGate>>(module, "SWAPGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SGate, quafu::BasicGate, std::shared_ptr<quafu::SGate>>(module, "SGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SdagGate, quafu::BasicGate, std::shared_ptr<quafu::SdagGate>>(module, "SdagGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SXGate, quafu::BasicGate, std::shared_ptr<quafu::SXGate>>(module, "SXGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SXdagGate, quafu::BasicGate, std::shared_ptr<quafu::SXdagGate>>(module, "SXdagGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::TGate, quafu::BasicGate, std::shared_ptr<quafu::TGate>>(module, "TGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::TdagGate, quafu::BasicGate, std::shared_ptr<quafu::TdagGate>>(module, "TdagGate")
        .def(py::init<const qbits_t &, const qbits_t &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::PauliChannel, quafu::BasicGate, std::shared_ptr<quafu::PauliChannel>>(module, "PauliChannel")
        .def(py::init<double, double, double, const qbits_t &, const qbits_t &>(), "px"_a, "py"_a, "pz"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::GroupedPauliChannel, quafu::BasicGate, std::shared_ptr<quafu::GroupedPauliChannel>>(
        module, "GroupedPauliChannel")
        .def(py::init<const quafu::VVT<double> &, const qbits_t &, const qbits_t &>(), "probs"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::DepolarizingChannel, quafu::BasicGate, std::shared_ptr<quafu::DepolarizingChannel>>(
        module, "DepolarizingChannel")
        .def(py::init<double, const qbits_t &, const qbits_t &>(), "p"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::AmplitudeDampingChannel, quafu::BasicGate, std::shared_ptr<quafu::AmplitudeDampingChannel>>(
        module, "AmplitudeDampingChannel")
        .def(py::init<bool, double, const qbits_t &, const qbits_t &>(), "daggered"_a, "damping_coeff"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::PhaseDampingChannel, quafu::BasicGate, std::shared_ptr<quafu::PhaseDampingChannel>>(
        module, "PhaseDampingChannel")
        .def(py::init<double, const qbits_t &, const qbits_t &>(), "damping_coeff"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::ThermalRelaxationChannel, quafu::BasicGate, std::shared_ptr<quafu::ThermalRelaxationChannel>>(
        module, "ThermalRelaxationChannel")
        .def(py::init<double, double, double, const qbits_t &, const qbits_t &>(), "t1"_a, "t2"_a, "gate_time"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
}

void BindTypeDependentGate(py::module &module) {  // NOLINT(runtime/references)
    using parameter::ParameterResolver;
    using quafu::CT;
    using quafu::Index;
    using quafu::qbits_t;
    using quafu::VT;
    using quafu::VVT;
    py::class_<quafu::RXGate, quafu::BasicGate, std::shared_ptr<quafu::RXGate>>(module, "RXGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RYGate, quafu::BasicGate, std::shared_ptr<quafu::RYGate>>(module, "RYGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RZGate, quafu::BasicGate, std::shared_ptr<quafu::RZGate>>(module, "RZGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RxxGate, quafu::BasicGate, std::shared_ptr<quafu::RxxGate>>(module, "RxxGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RyyGate, quafu::BasicGate, std::shared_ptr<quafu::RyyGate>>(module, "RyyGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RzzGate, quafu::BasicGate, std::shared_ptr<quafu::RzzGate>>(module, "RzzGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RxyGate, quafu::BasicGate, std::shared_ptr<quafu::RxyGate>>(module, "RxyGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RxzGate, quafu::BasicGate, std::shared_ptr<quafu::RxzGate>>(module, "RxzGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RyzGate, quafu::BasicGate, std::shared_ptr<quafu::RyzGate>>(module, "RyzGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::GivensGate, quafu::BasicGate, std::shared_ptr<quafu::GivensGate>>(module, "GivensGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::RotPauliString, quafu::BasicGate, std::shared_ptr<quafu::RotPauliString>>(module,
                                                                                                "RotPauliString")
        .def(py::init<const std::string &, const ParameterResolver &, const qbits_t &, const qbits_t &>(),
             "pauli_string"_a, "pr"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::SWAPalphaGate, quafu::BasicGate, std::shared_ptr<quafu::SWAPalphaGate>>(module, "SWAPalphaGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::GPGate, quafu::BasicGate, std::shared_ptr<quafu::GPGate>>(module, "GPGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::PSGate, quafu::BasicGate, std::shared_ptr<quafu::PSGate>>(module, "PSGate")
        .def(py::init<const ParameterResolver &, const qbits_t &, const qbits_t &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::U3, quafu::BasicGate, std::shared_ptr<quafu::U3>>(module, "u3")
        .def(py::init<const ParameterResolver &, const ParameterResolver &, const ParameterResolver &, const qbits_t &,
                      const qbits_t &>(),
             "theta"_a, "phi"_a, "lambda"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::Rn, quafu::BasicGate, std::shared_ptr<quafu::Rn>>(module, "rn")
        .def(py::init<const ParameterResolver &, const ParameterResolver &, const ParameterResolver &, const qbits_t &,
                      const qbits_t &>(),
             "alpha"_a, "beta"_a, "gamma"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::FSim, quafu::BasicGate, std::shared_ptr<quafu::FSim>>(module, "fsim")
        .def(py::init<const ParameterResolver &, const ParameterResolver &, const qbits_t &, const qbits_t &>(),
             "theta"_a, "phi"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::KrausChannel, quafu::BasicGate, std::shared_ptr<quafu::KrausChannel>>(module, "KrausChannel")
        .def(py::init<const VT<VVT<CT<double>>> &, const qbits_t &, const qbits_t &>(), "kraus_operator_set"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<quafu::CustomGate, quafu::BasicGate, std::shared_ptr<quafu::CustomGate>>(module, "CustomGate")
        .def(
            py::init<std::string, uint64_t, uint64_t, int, const ParameterResolver, const qbits_t &, const qbits_t &>(),
            "name"_a, "m_addr"_a, "dm_addr"_a, "dim"_a, "pr"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>())
        .def(py::init<std::string, const tensor::Matrix &, const qbits_t &, const qbits_t &>(), "name"_a, "mat"_a,
             "obj_qubits"_a, "ctrl_qubits"_a);
    py::class_<quafu::CustomTwoParamGate, quafu::BasicGate, std::shared_ptr<quafu::CustomTwoParamGate>>(
        module, "CustomTwoParamGate")
        .def(py::init<std::string, uint64_t, uint64_t, uint64_t, int, const ParameterResolver, const ParameterResolver,
                      const qbits_t &, const qbits_t &>(),
             "name"_a, "m_addr"_a, "dm_addr1"_a, "dm_addr2"_a, "dim"_a, "pr1"_a, "pr2"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
}
template <typename T>
auto BindOther(py::module &module) {
    using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
    using parameter::ParameterResolver;
    using quafu::CT;
    using quafu::Hamiltonian;
    using quafu::Index;
    using quafu::PauliTerm;
    using quafu::VS;
    using quafu::VT;
    using quafu::VVT;
    using quafu::python::CsrHdMatrix;
    // matrix

    // parameter resolver
    using quafu::sparse::Csr_Plus_Csr;
    using quafu::sparse::GetPauliMat;
    using quafu::sparse::PauliMat;
    using quafu::sparse::PauliMatToCsrHdMatrix;
    using quafu::sparse::SparseHamiltonian;
    using quafu::sparse::TransposeCsrHdMatrix;
    // pauli mat
    py::class_<PauliMat<T>, std::shared_ptr<PauliMat<T>>>(module, "pauli_mat")
        .def(py::init<>())
        .def(py::init<const PauliTerm<T> &, Index>())
        .def_readonly("n_qubits", &PauliMat<T>::n_qubits_)
        .def_readonly("dim", &PauliMat<T>::dim_)
        .def_readwrite("coeff", &PauliMat<T>::p_)
        .def("PrintInfo", &PauliMat<T>::PrintInfo);

    module.def("get_pauli_mat", &GetPauliMat<T>);

    // // csr_hd_matrix
    py::class_<CsrHdMatrix<T>, std::shared_ptr<CsrHdMatrix<T>>>(module, "csr_hd_matrix")
        .def(py::init<>())
        .def(py::init<Index, Index, py::array_t<Index>, py::array_t<Index>, py::array_t<CT<T>>>())
        .def("PrintInfo", &CsrHdMatrix<T>::PrintInfo);
    module.def("csr_plus_csr", &Csr_Plus_Csr<T>);
    module.def("transpose_csr_hd_matrix", &TransposeCsrHdMatrix<T>);
    module.def("pauli_mat_to_csr_hd_matrix", &PauliMatToCsrHdMatrix<T>);

    // hamiltonian
    py::class_<Hamiltonian<T>, std::shared_ptr<Hamiltonian<T>>>(module, "hamiltonian")
        .def(py::init<>())
        .def(py::init<const VT<PauliTerm<T>> &>())
        .def(py::init<const VT<PauliTerm<T>> &, Index>())
        .def(py::init<std::shared_ptr<CsrHdMatrix<T>>, Index>())
        .def_readwrite("how_to", &Hamiltonian<T>::how_to_)
        .def_readwrite("n_qubits", &Hamiltonian<T>::n_qubits_)
        .def_readwrite("ham", &Hamiltonian<T>::ham_)
        .def_readwrite("ham_sparse_main", &Hamiltonian<T>::ham_sparse_main_)
        .def_readwrite("ham_sparse_second", &Hamiltonian<T>::ham_sparse_second_);
    module.def("sparse_hamiltonian", &SparseHamiltonian<T>);
}
}  // namespace quafu::python

// Interface with python
PYBIND11_MODULE(quafubackend, m) {
    m.doc() = "quafu C++ plugin";

    py::module logging = m.def_submodule("logging", "quafu-C++ logging module");
    quafu::python::init_logging(logging);

    auto gate_id = py::enum_<quafu::GateID>(m, "GateID")
                       .value("I", quafu::GateID::I)
                       .value("X", quafu::GateID::X)
                       .value("Y", quafu::GateID::Y)
                       .value("Z", quafu::GateID::Z)
                       .value("RX", quafu::GateID::RX)
                       .value("RY", quafu::GateID::RY)
                       .value("RZ", quafu::GateID::RZ)
                       .value("Rxx", quafu::GateID::Rxx)
                       .value("Ryy", quafu::GateID::Ryy)
                       .value("Rzz", quafu::GateID::Rzz)
                       .value("Givens", quafu::GateID::Givens)
                       .value("Rn", quafu::GateID::Rn)
                       .value("H", quafu::GateID::H)
                       .value("SWAP", quafu::GateID::SWAP)
                       .value("ISWAP", quafu::GateID::ISWAP)
                       .value("SWAPalpha", quafu::GateID::SWAPalpha)
                       .value("T", quafu::GateID::T)
                       .value("Tdag", quafu::GateID::Tdag)
                       .value("S", quafu::GateID::S)
                       .value("Sdag", quafu::GateID::Sdag)
                       .value("SX", quafu::GateID::SX)
                       .value("SXdag", quafu::GateID::SXdag)
                       .value("CNOT", quafu::GateID::CNOT)
                       .value("CZ", quafu::GateID::CZ)
                       .value("PauliString", quafu::GateID::PauliString)
                       .value("RotPauliString", quafu::GateID::RPS)
                       .value("GP", quafu::GateID::GP)
                       .value("PS", quafu::GateID::PS)
                       .value("U3", quafu::GateID::U3)
                       .value("FSim", quafu::GateID::FSim)
                       .value("M", quafu::GateID::M)
                       .value("PL", quafu::GateID::PL)
                       .value("DEP", quafu::GateID::DEP)
                       .value("AD", quafu::GateID::AD)
                       .value("PD", quafu::GateID::PD)
                       .value("KRAUS", quafu::GateID::KRAUS)
                       .value("TR", quafu::GateID::TR)
                       .value("CUSTOM", quafu::GateID::CUSTOM)
                       .value("CUSTOM_TWO_PARAM", quafu::GateID::CUSTOM_TWO_PARAM);
    gate_id.attr("__repr__") = pybind11::cpp_function(
        [](const quafu::GateID &id) -> pybind11::str { return fmt::format("GateID.{}", id); }, pybind11::name("name"),
        pybind11::is_method(gate_id));
    gate_id.attr("__str__") = pybind11::cpp_function(
        [](const quafu::GateID &id) -> pybind11::str { return fmt::format("{}", id); }, pybind11::name("name"),
        pybind11::is_method(gate_id));

    m.attr("EQ_TOLERANCE") = py::float_(1.e-8);

    py::module gate = m.def_submodule("gate", "quafu-C++ gate");
    py::class_<quafu::BasicGate, std::shared_ptr<quafu::BasicGate>>(gate, "BasicGate")
        .def(py::init<quafu::GateID, const quafu::qbits_t &, const quafu::qbits_t &>())
        .def("get_id", &quafu::BasicGate::GetID)
        .def("get_obj_qubits", &quafu::BasicGate::GetObjQubits)
        .def("get_ctrl_qubits", &quafu::BasicGate::GetCtrlQubits);
    quafu::python::BindTypeIndependentGate(gate);
    quafu::python::BindTypeDependentGate(gate);

    py::module quafubackend_double = m.def_submodule("double", "quafu-C++ double backend");
    quafu::python::BindOther<double>(quafubackend_double);
    py::module quafubackend_float = m.def_submodule("float", "quafu-C++ double backend");
    quafu::python::BindOther<float>(quafubackend_float);

    py::module c = m.def_submodule("c", "pybind11 c++ env");
    quafu::BindPybind11Env(c);

    py::module device = m.def_submodule("device", "Quantum device module");
    quafu::python::BindTopology(device);
    quafu::python::BindQubitMapping(device);
}

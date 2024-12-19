/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "config/openmp.h"
#include "config/type_promotion.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/vector/detail/cpu_vector_policy.h"
namespace quafu::sim::vector::detail {
constexpr int ROT_PAULI_FACTOR = 2;
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR)) * IMAGE_MI;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR) * IMAGE_MI;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] + s * qs[k];
                auto v10 = c * qs[k] + s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] + s * qs[m];
                    auto v01 = c * qs[j] + s * qs[k];
                    auto v10 = c * qs[k] + s * qs[j];
                    auto v11 = c * qs[m] + s * qs[i];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] - s * qs[m];
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = c * qs[k] + s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] - s * qs[m];
                    auto v01 = c * qs[j] - s * qs[k];
                    auto v10 = c * qs[k] + s * qs[j];
                    auto v11 = c * qs[m] + s * qs[i];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR)) * IMAGE_MI;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR) * IMAGE_MI;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] + s * qs[j];
                auto v01 = c * qs[j] + s * qs[i];
                auto v10 = c * qs[k] - s * qs[m];
                auto v11 = c * qs[m] - s * qs[k];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] + s * qs[j];
                    auto v01 = c * qs[j] + s * qs[i];
                    auto v10 = c * qs[k] - s * qs[m];
                    auto v11 = c * qs[m] - s * qs[k];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs,
                                                            const qbits_t& ctrls, calc_type val, index_t dim,
                                                            bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val);
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val);
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = s * qs[j] + c * qs[k];
                qs[j] = v01;
                qs[k] = v10;
                if (diff) {
                    qs[i] = 0.0;
                    qs[m] = 0.0;
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v01 = c * qs[j] - s * qs[k];
                    auto v10 = s * qs[j] + c * qs[k];
                    qs[j] = v01;
                    qs[k] = v10;
                    if (diff) {
                        qs[i] = 0.0;
                        qs[m] = 0.0;
                    }
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] - s * qs[j];
                auto v01 = c * qs[j] + s * qs[i];
                auto v10 = c * qs[k] + s * qs[m];
                auto v11 = c * qs[m] - s * qs[k];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] - s * qs[j];
                    auto v01 = c * qs[j] + s * qs[i];
                    auto v10 = c * qs[k] + s * qs[m];
                    auto v11 = c * qs[m] - s * qs[k];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR)) * IMAGE_I;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR) * IMAGE_I;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = c * qs[k] - s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] + s * qs[m];
                    auto v01 = c * qs[j] - s * qs[k];
                    auto v10 = c * qs[k] - s * qs[j];
                    auto v11 = c * qs[m] + s * qs[i];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR));
    auto s = static_cast<calc_type_>(std::sin(val / ROT_PAULI_FACTOR));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
        s = static_cast<calc_type_>(std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR);
    }
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                index_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                qs[i] *= me;
                qs[j] *= e;
                qs[k] *= e;
                qs[m] *= me;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                index_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    qs[i] *= me;
                    qs[j] *= e;
                    qs[k] *= e;
                    qs[m] *= me;
                }
            })
        if (diff) {
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / ROT_PAULI_FACTOR);
    auto b = -std::sin(val / ROT_PAULI_FACTOR);
    if (diff) {
        a = -std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
        b = -std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / ROT_PAULI_FACTOR);
    auto b = std::sin(val / ROT_PAULI_FACTOR);
    if (diff) {
        a = -std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
        b = std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRPS(qs_data_p_t* qs_p, const PauliMask& mask, Index ctrl_mask,
                                                         calc_type val, index_t dim, bool diff) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto a = std::cos(val / ROT_PAULI_FACTOR);
    auto b = std::sin(val / ROT_PAULI_FACTOR);
    if (diff) {
        a = -std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
        b = std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    if (ctrl_mask == 0) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = i ^ mask_f;
                if (i <= j) {
                    auto axis2power = CountOne(i & mask.mask_z);  // -1
                    auto axis3power = CountOne(i & mask.mask_y);  // -1j
                    auto c = ComplexCast<double, calc_type>::apply(
                        POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    if (i == j) {
                        qs[i] *= (a - IMAGE_I * b * c);
                    } else {
                        auto qs_i = qs[i];
                        auto qs_j = qs[j];
                        qs[i] = qs_i * a - IMAGE_I * qs_j * b / c;
                        qs[j] = qs_j * a - IMAGE_I * qs_i * b * c;
                    }
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                if ((i & ctrl_mask) == ctrl_mask) {
                    auto j = i ^ mask_f;
                    if (i <= j) {
                        auto axis2power = CountOne(i & mask.mask_z);  // -1
                        auto axis3power = CountOne(i & mask.mask_y);  // -1j
                        auto c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                        if (i == j) {
                            qs[i] *= (a - IMAGE_I * b * c);
                        } else {
                            auto qs_i = qs[i];
                            auto qs_j = qs[j];
                            qs[i] = qs_i * a - IMAGE_I * qs_j * b / c;
                            qs[j] = qs_j * a - IMAGE_I * qs_i * b * c;
                        }
                    }
                }
            })
    }
    if (diff && ctrl_mask) {
        derived::SetToZeroExcept(qs_p, ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / ROT_PAULI_FACTOR);
    auto b = std::sin(val / ROT_PAULI_FACTOR);
    if (diff) {
        a = -std::sin(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
        b = std::cos(val / ROT_PAULI_FACTOR) / ROT_PAULI_FACTOR;
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace quafu::sim::vector::detail

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
#include <cmath>

#include "config/openmp.h"
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
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                          index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto tmp = qs[j];
                qs[j] = qs[k];
                qs[k] = tmp;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto tmp = qs[j];
                    qs[j] = qs[k];
                    qs[k] = tmp;
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                           bool daggered, index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto tmp = qs[i + mask.obj_min_mask];
                qs[i + mask.obj_min_mask] = frac * qs[i + mask.obj_max_mask] * IMAGE_I;
                qs[i + mask.obj_max_mask] = frac * tmp * IMAGE_I;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto tmp = qs[i + mask.obj_min_mask];
                    qs[i + mask.obj_min_mask] = frac * qs[i + mask.obj_max_mask] * IMAGE_I;
                    qs[i + mask.obj_max_mask] = frac * tmp * IMAGE_I;
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = std::exp(IMAGE_I * static_cast<calc_type_>(M_PI) * val);
    auto a = (static_cast<calc_type_>(1) + e) / static_cast<calc_type_>(2);
    auto b = (static_cast<calc_type_>(1) - e) / static_cast<calc_type_>(2);
    if (!diff) {
        if (!mask.ctrl_mask) {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    omp::idx_t i;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, i);
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto tmp_j = qs[j];
                    auto tmp_k = qs[k];
                    qs[j] = a * tmp_j + b * tmp_k;
                    qs[k] = b * tmp_j + a * tmp_k;
                })
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    omp::idx_t i;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, i);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto m = i + mask.obj_mask;
                        auto j = i + mask.obj_min_mask;
                        auto k = i + mask.obj_max_mask;
                        auto tmp_j = qs[j];
                        auto tmp_k = qs[k];
                        qs[j] = a * tmp_j + b * tmp_k;
                        qs[k] = b * tmp_j + a * tmp_k;
                    }
                })
        }
    } else {
        a = IMAGE_I * static_cast<calc_type_>(M_PI_2) * e;
        b = IMAGE_MI * static_cast<calc_type_>(M_PI_2) * e;
        if (!mask.ctrl_mask) {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    omp::idx_t i;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, i);
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto tmp_j = qs[j];
                    auto tmp_k = qs[k];
                    qs[i] = 0;
                    qs[m] = 0;
                    qs[j] = a * tmp_j + b * tmp_k;
                    qs[k] = b * tmp_j + a * tmp_k;
                })
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    omp::idx_t i;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, i);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto m = i + mask.obj_mask;
                        auto j = i + mask.obj_min_mask;
                        auto k = i + mask.obj_max_mask;
                        auto tmp_j = qs[j];
                        auto tmp_k = qs[k];
                        qs[i] = 0;
                        qs[m] = 0;
                        qs[j] = a * tmp_j + b * tmp_k;
                        qs[k] = b * tmp_j + a * tmp_k;
                    }
                })
            derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
        }
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

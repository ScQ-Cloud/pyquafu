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
#include "config/openmp.hpp"

#include "core/parameter_resolver.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/cpu_vector_policy.hpp"

namespace mindquantum::sim::vector::detail {
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
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
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
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

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 bool daggered, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto tmp = qs[i + mask.obj_min_mask];
                qs[i + mask.obj_min_mask] = frac * qs[i + mask.obj_max_mask] * IMAGE_I;
                qs[i + mask.obj_max_mask] = frac * tmp * IMAGE_I;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
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

template struct CPUVectorPolicyBase<float>;
template struct CPUVectorPolicyBase<double>;

}  // namespace mindquantum::sim::vector::detail

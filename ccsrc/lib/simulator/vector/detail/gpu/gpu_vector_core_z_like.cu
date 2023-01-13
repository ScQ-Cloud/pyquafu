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
#include "simulator/vector/detail/gpu_vector_policy.cuh"

namespace mindquantum::sim::vector::detail {
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyZLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 qs_data_t val, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask) + obj_mask;
            qs[i] *= val;
        });
    } else {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask) + obj_mask;
            if ((i & ctrl_mask) == ctrl_mask) {
                qs[i] *= val;
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, -1, dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySGate(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, 1), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, -1), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, 1) / std::sqrt(2.0), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyTdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, -1) / std::sqrt(2.0), dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyPS(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    if (!diff) {
        ApplyZLike(qs, objs, ctrls, qs_data_t(std::cos(val), std::sin(val)), dim);
    } else {
        SingleQubitGateMask mask(objs, ctrls);
        thrust::counting_iterator<index_t> l(0);
        auto obj_high_mask = mask.obj_high_mask;
        auto obj_low_mask = mask.obj_low_mask;
        auto obj_mask = mask.obj_mask;
        auto ctrl_mask = mask.ctrl_mask;
        qs_data_t e = qs_data_t(-std::sin(val), std::cos(val));
        if (!mask.ctrl_mask) {
            if (!mask.ctrl_mask) {
                thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
                    auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                    auto j = i + obj_mask;
                    qs[i] = 0;
                    qs[j] *= e;
                });
            } else {
                thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
                    auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                    if ((i & ctrl_mask) == ctrl_mask) {
                        auto j = i + obj_mask;
                        qs[i] = 0;
                        qs[j] *= e;
                    }
                });
            }
            GPUVectorPolicyBase::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template struct GPUVectorPolicyBase<float>;
template struct GPUVectorPolicyBase<double>;

}  // namespace mindquantum::sim::vector::detail

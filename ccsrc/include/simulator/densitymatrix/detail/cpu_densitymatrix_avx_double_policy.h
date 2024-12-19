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
#ifndef INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_AVX_DOUBLE_POLICY_HPP
#define INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_AVX_DOUBLE_POLICY_HPP
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace quafu::sim::densitymatrix::detail {
struct CPUDensityMatrixPolicyAvxDouble : public CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double> {};
}  // namespace quafu::sim::densitymatrix::detail
#endif

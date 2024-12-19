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

#ifndef QUAFU_CATCH2_STD_COMPLEX_HPP
#define QUAFU_CATCH2_STD_COMPLEX_HPP

#include <complex>

#include "quafu/catch2/catch2_fmt_formatter.hpp"

#include <catch2/catch.hpp>

// =============================================================================

namespace Catch {
template <typename float_t>
struct StringMaker<std::complex<float_t>> : quafu::catch2::FmtStringMakerBase<std::complex<float>> {};
}  // namespace Catch

// =============================================================================

#endif /* QUAFU_CATCH2_STD_COMPLEX_HPP */

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

#include "core/utils.h"

#include "core/quafu_base_types.h"

namespace quafu {
const VT<CT<double>> POLAR = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
TimePoint NOW() {
    return std::chrono::steady_clock::now();
}

int TimeDuration(TimePoint start, TimePoint end) {
    auto d = end - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

void safe_copy(void *dest, size_t dest_size, const void *src, size_t count) {
    if ((dest == NULL && dest_size != 0) || (src == NULL && count != 0)) {
        throw std::runtime_error("Invalid parameters for safe_memcpy.");
    }
    if (count > dest_size) {
        throw std::runtime_error("Buffer overflow in safe_memcpy.");
    }

    unsigned char *pDest = reinterpret_cast<unsigned char *>(dest);
    const unsigned char *pSrc = reinterpret_cast<const unsigned char *>(src);

    for (size_t i = 0; i < count; i++) {
        pDest[i] = pSrc[i];
    }
}

Index GetControlMask(const qbits_t &ctrls) {
    Index ctrlmask = std::accumulate(ctrls.begin(), ctrls.end(), 0,
                                     [&](Index a, Index b) { return a | (static_cast<uint64_t>(1) << b); });
    return ctrlmask;
}

PauliMask GetPauliMask(const VT<PauliWord> &pws) {
    VT<Index> out = {0, 0, 0, 0, 0, 0};
    for (auto &pw : pws) {
        for (Index i = 0; i < 3; i++) {
            if (static_cast<Index>(pw.second - 'X') == i) {
                out[i] += (static_cast<uint64_t>(1) << pw.first);
                out[3 + i] += 1;
            }
        }
    }
    PauliMask res = {out[0], out[1], out[2], out[3], out[4], out[5]};
    return res;
}
}  // namespace quafu

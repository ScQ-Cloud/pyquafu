/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "simulator/vector/runtime/utils.h"

namespace quafu::sim::rt {
std::tuple<bool, Index> convert_int(const std::string &s, int64_t limit, bool raise_error) {
    char *p = nullptr;
    Index converted = strtol(s.c_str(), &p, 10);
    if (*p) {
        if (raise_error) {
            throw std::runtime_error("Cannot convert '" + s + "' to number.");
        }
        return {false, 0};
    }
    if (converted > limit) {
        if (raise_error) {
            throw std::runtime_error("Number " + std::to_string(converted) + " to large, limit is "
                                     + std::to_string(limit));
        }
        return {false, converted};
    }
    return {true, converted};
}

std::tuple<bool, double> convert_double(const std::string &s, bool raise_error) {
    char *p = nullptr;
    double converted = strtod(s.c_str(), &p);
    if (*p) {
        if (raise_error) {
            throw std::runtime_error("Cannot convert '" + s + "' to double.");
        }
        return {false, 0};
    }
    return {true, converted};
}
}  // namespace quafu::sim::rt

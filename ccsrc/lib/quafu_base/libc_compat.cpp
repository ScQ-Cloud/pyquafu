/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifdef __linux__
#    include "config/libc_compat.h"

#    include <features.h>
#    include <math.h>
double __exp_finite(double x) {
    return exp(x);
}
float __expf_finite(float x) {
    return expf(x);
}
double __pow_finite(double x, double y) {
    return pow(x, y);
}
#endif

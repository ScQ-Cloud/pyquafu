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

#include "config/logging.h"

#include <string>

#include <spdlog/sinks/basic_file_sink.h>

namespace quafu::logging {
void set_log_file(const std::string& filename) {
    spdlog::set_default_logger(spdlog::basic_logger_mt("default_log", filename));
    QUAFU_INFO("Setting log file to {}", filename);
}
}  // namespace quafu::logging

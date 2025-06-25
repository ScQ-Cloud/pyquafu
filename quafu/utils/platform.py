# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from ..exceptions import QuafuError


def get_homedir():
    if sys.platform == "win32":
        return os.environ["USERPROFILE"]
    if sys.platform in ["darwin", "linux"]:
        return os.environ["HOME"]
    raise QuafuError(
        f"unsupported platform:{sys.platform}. You may raise a request issue on github."
    )

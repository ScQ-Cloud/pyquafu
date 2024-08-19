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

from typing import List
from quafu.transpiler.passes.mapping.baselayout import Layout


class DataDict(dict):
    """A default dictionary-like object"""

    def __init__(self, *args, **kwargs):
        super(DataDict, self).__init__(*args, **kwargs)

        self['coupling_list']: List = None
        self['initial_layout']: Layout = None
        self['final_layout']: Layout = None
        self['variables']: List = None

    def __missing__(self, key):
        return None

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

from typing import List, Dict

from quafu.transpiler.backends.backend import Backend
from quafu.transpiler.passes.datadict import DataDict

class Model:
    """
    Represents a model of a quantum system.

    Attributes:
        _backend (Backend): The backend of the quantum system.
        _layout (List[int]): Layout of the quantum system.
    """

    # def __init__(self, backend: Backend, layout: Optional[Layout] = None):
    #     self._backend = backend
    #     self._layout = layout

    def __init__(self, backend: Backend, layout: Dict = None, datadict: DataDict = None):
        self._backend = backend
        self._layout = {'initial_layout': None, 'final_layout': None}
        self.set_layout(layout)
        self.datadict = datadict

    def get_backend(self) -> Backend:
        """Return the backend of the model."""
        return self._backend

    # def set_layout(self, layout: List[int]):
    #     """Set a new layout for the quantum system."""
    #     self._layout = layout

    def set_layout(self, new_layout: Dict):
        """Set a new layout for the quantum system."""
        if isinstance(new_layout, Dict):
            if 'initial_layout' in new_layout.keys():
                self._layout['initial_layout'] = new_layout['initial_layout']
            if 'final_layout' in new_layout.keys():
                self._layout['final_layout'] = new_layout['final_layout']

    def get_layout(self) -> Dict:
        """Return the layout of the quantum system."""
        return self._layout

    def get_datadict(self) -> Dict:
        """Return the datadict of the quantum system."""
        return self.datadict

    def __repr__(self):
        return f"<Model(backend={self.get_backend()}, layout={self.get_layout()})>"

    def backend_info(self) -> str:
        """Return a string representation of the backend's information."""
        return repr(self._backend)

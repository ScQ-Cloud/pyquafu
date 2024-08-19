from typing import List, Dict

from quafu.transpile.backends.backend import Backend
from quafu.transpile.passes.datadict import DataDict
from quafu.transpile.passflow.preset_passflow import PresetPassflow

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

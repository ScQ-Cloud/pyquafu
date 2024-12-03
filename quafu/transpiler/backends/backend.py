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


from typing import Dict, List, Optional, Union

from quafu.transpiler.graph.couplinggraph import CouplingGraph


class Backend:
    """
    Represents the backend of a quantum system.

    Args:
        name (str): The name of the backend.
        backend_type (str): The type of the backend (e.g., superconducting, ion trap, atomic).
        qubits_num (int): Number of qubits in the backend.
        coupling_map (List[Union[int, tuple]]): The coupling map for the qubits.
        T1 (Optional[float]): The T1 time.
        T2 (Optional[float]): The T2 time.
        fidelity (Optional[float]): Fidelity of the backend.
        basis_gates (Optional[List[str]]): List of basis gates supported.
        pulse (Optional[bool]): Indicates if the backend supports pulse level operations.
        calibration_time (Optional[str]): The calibration time.

    Attributes:
        _properties (dict): A dictionary containing all the properties of the backend.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        name: str = "backend",
        backend_type: str = "superconducting",
        qubits_num: int = None,
        qubits_info: List[Dict] = None,
        coupling_list: List[Union[int, tuple]] = None,
        coupling_graph: CouplingGraph = None,
        avg_t1: Optional[float] = None,
        avg_t2: Optional[float] = None,
        avg_fidelity: Optional[float] = None,
        basis_gates: Optional[List[str]] = None,
        pulse: bool = False,
        calibration_time: Optional[float] = None,
        status: str = None,
    ):
        # Setting up bidirectional coupling graph.
        if coupling_list is not None:
            coupling_graph = CouplingGraph(coupling_list)
            if coupling_graph.is_bidirectional is False:
                coupling_graph.do_bidirectional()
        elif coupling_graph is not None:
            if coupling_graph.is_bidirectional is False:
                coupling_graph.do_bidirectional()
        else:
            print(
                "Warning: Both coupling_list and coupling_graph are None, "
                "which means there is no qubits coupling structure."
            )

        self._properties = {
            "name": name,
            "backend_type": backend_type,
            "qubits_num": qubits_num,
            "qubits_info": qubits_info,
            "coupling_list": coupling_list,
            "coupling_graph": coupling_graph,
            "avg_t1": avg_t1,
            "avg_t2": avg_t2,
            "avg_fidelity": avg_fidelity,
            "basis_gates": basis_gates,
            "pulse": pulse,
            "calibration_time": calibration_time,
            "status": status,
        }

    def get_all_properties(self):
        """get all the properties of the backend."""
        return self._properties

    def get_property(self, property_name: str) -> Union[str, int, float, List, None]:
        """Return the value of the specified property."""
        return self._properties.get(property_name, None)

    def set_property(self, property_name: str, value: Union[str, int, float, List]):
        """Set a new value for the specified property."""
        if property_name in self._properties:
            self._properties[property_name] = value

    def __repr__(self):
        return (
            f"<Backend(name={self.get_property('name')}, "
            f"type={self.get_property('backend_type')}, qubits_num={self.get_property('qubits_num')})>"
        )

    def supports_pulse(self):
        """Check if the backend supports pulse level operations."""
        return self.get_property("pulse")

    def available_basis_gates(self) -> Optional[List[str]]:
        """Return the list of basis gates supported by the backend."""
        return self.get_property("basis_gates")

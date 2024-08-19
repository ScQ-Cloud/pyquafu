from typing import List, Union

from quafu import QuantumCircuit

from quafu.dagcircuits.circuit_dag import dag_to_circuit
from quafu.dagcircuits.dag_circuit import DAGCircuit
from quafu.transpile.passes.basepass import BasePass
from quafu.transpile.passes.datadict import DataDict
from quafu.transpile.passflow.preset_passflow import PresetPassflow


class Compiler:
    def __init__(self, workflow: List[BasePass] = None, initial_model=None):
        self.workflow = workflow
        if initial_model is None:
            self.model = DataDict()
        else:
            self.model = initial_model
            if self.model.datadict is None:
                self.model.datadict = DataDict()

    def set_model(self, new_model):
        self.model = new_model

    def compile(self, circuit: Union[QuantumCircuit, DAGCircuit], optimization_level: int = 0):
        # give the parameters of the original circuit only once,be careful!
        self.model.datadict['variables'] = circuit.variables


        if self.workflow is None:
            if optimization_level in [0, 1, 2, 3]:
                self.workflow = PresetPassflow(self.model._backend.get_property('basis_gates'),
                                               optimization_level=optimization_level).get_passflow()

            # if optimization_level == 0:
            #     pass
            # elif optimization_level == 1:
            #     pass
            # elif optimization_level == 2:
            #     pass
            else:
                raise ValueError("Error: The value of optimization_level is between [0,3].")

        for pass_instance in self.workflow:
            if hasattr(pass_instance, 'set_model'):
                pass_instance.set_model(self.model)

            circuit = pass_instance.run(circuit)

            if hasattr(pass_instance, 'get_model'):
                self.model = pass_instance.get_model()

        if isinstance(circuit, DAGCircuit):
            circuit = dag_to_circuit(circuit, circuit.circuit_qubits)

        return circuit

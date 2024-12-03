import math

import numpy as np
from quafu.elements.element_gates import (
    CRYGate,
    CRZGate,
    CXGate,
    HGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZZGate,
    U3Gate,
    XGate,
)
from quafu.elements.parameters import Parameter

from quafu import QuantumCircuit, simulate


class TestBuildingCircuit:
    def test_control(self):
        q = QuantumCircuit(3)
        q << (XGate(1))
        q << (CXGate(0, 2))
        q << (RXGate(0, 0.1))
        q << (PhaseGate(2, 1.0)).ctrl_by([0])
        q << RZZGate(0, 2, 0.26)
        nq = q.add_controls(2, inplace=False)
        nq.draw_circuit()
        try:
            nq.to_openqasm()
        except NotImplementedError as e:
            print(e)

    def test_join(self):
        q = QuantumCircuit(3)
        q << (XGate(1))
        q << (CXGate(0, 2))
        q1 = QuantumCircuit(2)
        q1 << HGate(1) << CXGate(1, 0)
        q.join(q1, [2, 3])
        q.draw_circuit()

    def test_power(self):
        q = QuantumCircuit(2)
        q << HGate(0)
        q << HGate(1)
        q << U3Gate(1, 0.2, 0.1, 0.3)
        q << U3Gate(1, 0.2, 0.1, 0.3)
        q << RYYGate(0, 1, 0.4)
        q << RYYGate(0, 1, 0.4)
        q << RXGate(0, 0.2)
        q << RXGate(0, 0.2)
        q << CRYGate(0, 1, 0.23)
        q << CRYGate(0, 1, 0.23)
        q1 = QuantumCircuit(2)
        q1 << HGate(0)
        q1 << HGate(1)
        q1 << U3Gate(1, 0.2, 0.1, 0.3).power(2)
        q1 << RYYGate(0, 1, 0.4).power(2)
        q1 << RXGate(0, 0.2).power(2)
        q1 << CRYGate(0, 1, 0.23).power(2)

        sv1 = simulate(q).get_statevector()
        sv2 = simulate(q1).get_statevector()
        assert math.isclose(np.abs(np.dot(sv1, sv2.conj())), 1.0)

    def test_dagger(self):
        q = QuantumCircuit(3)
        q << HGate(0)
        q << HGate(1)
        q << HGate(2)
        q << RXGate(2, 0.3)
        q << RYGate(2, Parameter("p1", 0.1))
        q << CXGate(0, 1)
        q << CRZGate(2, 1, 0.2)
        q << RXXGate(0, 2, 1.2)

        q3 = QuantumCircuit(3)
        q3 << RXXGate(0, 2, -1.2)
        q3 << CRZGate(2, 1, -0.2)
        q3 << CXGate(0, 1)
        q3 << RYGate(2, -0.1)
        q3 << RXGate(2, -0.3)
        q3 << HGate(0)
        q3 << HGate(1)
        q3 << HGate(2)

        sv1 = simulate(q.dagger()).get_statevector()
        sv3 = simulate(q3).get_statevector()
        assert math.isclose(np.abs(np.dot(sv1, sv3.conj())), 1.0)

    def test_wrapper_power(self):
        q = QuantumCircuit(2)
        q << HGate(0)
        q << HGate(1)

        q1 = QuantumCircuit(2)
        q1 << U3Gate(1, 0.2, 0.1, 0.3)
        q1 << RYYGate(0, 1, Parameter("p1", 0.4))
        q1 << RXGate(0, 0.2)
        q1 << CRYGate(0, 1, 0.23)
        nq1 = q.join(q1.power(2), inplace=False)
        nq2 = q.join(q1.wrap().power(2), inplace=False)

        nq1.draw_circuit()
        nq2.unwrap().draw_circuit()
        from quafu.simulators.simulator import SVSimulator

        backend = SVSimulator()
        sv1 = backend.run(nq1)["statevector"]
        sv2 = backend.run(nq2)["statevector"]
        assert math.isclose(np.abs(np.dot(sv1, sv2.conj())), 1.0)

    def test_wrapper(self, inplace=True):
        q = QuantumCircuit(3)
        q << HGate(0)
        q << HGate(1)
        q << HGate(2)

        q1 = QuantumCircuit(3)
        q1 << CXGate(0, 1)
        q1 << RXGate(2, Parameter("p1", 0.2))

        q = q.join(q1.wrap(), qbits=[2, 3, 4], inplace=inplace)
        q.draw_circuit()
        from quafu.simulators.simulator import SVSimulator

        backend = SVSimulator()
        backend.run(q)["statevector"]
        q.unwrap().draw_circuit()

        # multi-wrapper
        q2 = QuantumCircuit(3)
        q2 << XGate(0)
        q2 << CXGate(0, 1)
        q2 = q2.join(q.wrap())
        q2.draw_circuit()
        sv2 = backend.run(q2)["statevector"]

        q2.unwrap()
        q2.draw_circuit()
        sv3 = backend.run(q2)["statevector"]
        assert math.isclose(np.abs(np.dot(sv2, sv3.conj())), 1.0)

    def test_control_wrapper(self):
        q = QuantumCircuit(3)
        q << HGate(0)
        q << HGate(1)
        q << HGate(2)

        q1 = QuantumCircuit(3)
        q1 << CRYGate(0, 1, Parameter("p1", 0.3))
        q1 << RXGate(2, 0.2)

        q.join(q1.wrap().add_controls([3, 4]), [2, 3, 4, 5, 6])
        q.draw_circuit()
        from quafu.simulators.simulator import SVSimulator

        backend = SVSimulator()
        sv1 = backend.run(q)["statevector"]
        q.unwrap()
        q.draw_circuit()
        sv2 = backend.run(q)["statevector"]
        assert math.isclose(np.abs(np.dot(sv1, sv2.conj())), 1.0)


if __name__ == "__main__":
    TestBuildingCircuit().test_wrapper()

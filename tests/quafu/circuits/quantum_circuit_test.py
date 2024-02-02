from quafu.circuits import QuantumCircuit
from quafu.elements.element_gates import RXGate
from quafu.elements.parameters import Parameter
import math


class TestQuantumCircuit:
    def test_parameterized_gates(self):
        """Test get parameterized gates"""
        c = QuantumCircuit(2)
        c.x(0)
        c.rx(1, 0.5)
        g = c.parameterized_gates[0]
        print("\n ------ Testing ------ \n")
        c.draw_circuit()
        assert isinstance(g, RXGate)
        assert math.isclose(g.paras[0], 0.5)

    def test_update_parameters(self):
        """Test parameter update"""
        c = QuantumCircuit(2)
        c.x(0)
        c.rx(1, 0.5)
        c.update_params([0.1])
        g = c.parameterized_gates[0]
        c.draw_circuit()
        assert isinstance(g, RXGate)
        assert math.isclose(g.paras[0], 0.1)
        c.update_params([0.2])
        assert math.isclose(g.paras[0], 0.2)
        c.update_params([None])
        assert math.isclose(g.paras[0], 0.2)

    def test_instantiated_params(self):
        """Create Parameter objects"""
        pq = QuantumCircuit(4)
        theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]

        for i in range(4):
            pq.rx(i, theta[i])

        pq.ry(2, theta[0]*theta[1]-3.*theta[0])
        pq.draw_circuit()

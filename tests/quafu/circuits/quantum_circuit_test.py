import math

from quafu.circuits import QuantumCircuit
from quafu.elements.element_gates import RXGate


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
        assert math.isclose(g.paras, 0.5)

    def test_update_parameters(self):
        """Test parameter update"""
        c = QuantumCircuit(2)
        c.x(0)
        c.rx(1, 0.5)
        c.update_params([0.1])
        g = c.parameterized_gates[0]
        c.draw_circuit()
        assert isinstance(g, RXGate)
        assert math.isclose(g.paras, 0.1)
        c.update_params([0.2])
        assert math.isclose(g.paras, 0.2)
        c.update_params([None])
        assert math.isclose(g.paras, 0.2)

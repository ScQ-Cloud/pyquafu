from quafu.circuits import QuantumCircuit

class TestQuantumCircuit:

    def test_parameterized_gates(self):
        """Test get parameterized gates"""
        c = QuantumCircuit(2)
        c.x(0)
        c.rx(1, 0.5)
        print(c.parameterized_gates)

from quafu.algorithms.ansatz import QAOACircuit


class TestQAOACircuit:
    def test_build(self):
        qaoa = QAOACircuit("IIZZ")
        qaoa.draw_circuit()

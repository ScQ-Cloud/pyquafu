

from quantum_circuit import QuantumCircuit
#from qcover import ...

class QAOACircuit(QuantumCircuit):
    def __init__(self, graph, paras):
        self.graph = graph
        self.paras = paras
        num = graph.num # number of vertex
        super.__init__(num)

    def gen_circuit(self):
        """circuit from qcover"""
        pass
    
    def upate_paras(self, paras):
        self.paras = paras
        self.gen_circuit()





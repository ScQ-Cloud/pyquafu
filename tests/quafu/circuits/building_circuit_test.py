from quafu import QuantumCircuit, simulate
from quafu.elements.element_gates import *
from quafu.elements.parameters import Parameter
import numpy as np
import math

class TestBuildingCircuit:
    #TODO:add wrapper test
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
        q1 =  QuantumCircuit(2)
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

        sv1 = simulate(q, output="state_vector").get_statevector()
        sv2 = simulate(q1, output="state_vector").get_statevector()
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

        sv1 = simulate(q.dagger(), output="state_vector").get_statevector()
        sv3 = simulate(q3, output="state_vector").get_statevector()
        assert math.isclose(np.abs(np.dot(sv1, sv3.conj())), 1.0)

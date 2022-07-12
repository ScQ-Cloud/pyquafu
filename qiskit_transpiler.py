#%%---------------
import numpy as np
from qiskit.visualization import plot_circuit_layout
from qiskit.quantum_info import random_unitary


from quantum_circuit import QuantumCircuit as mQC
from qiskit  import QuantumCircuit, transpile, QuantumRegister
from qiskit.test.mock import FakeBoeblingen

backend = FakeBoeblingen()

n = 5
couplings = [list(range(n)[i:i+2]) for i in range(n-1)]
couplings.extend([list(range(n)[i:i+2])[::-1] for i in range(n-1)])
qc = QuantumCircuit(5)
qc.h(2)
qc.h(3)     
qc.h(4)
trans_circ = transpile(qc,  backend, coupling_map=couplings, optimization_level=3)


qc.cx(3, 2)
qc.cx(1, 3)
qc.barrier([1, 2])
# qc.cx(1, 2)
qc.x(1)
qc.y(2)
qc.rz(0.1, 3)
qc.iswap(3, 4)
qc.h(3)

# qc.x(4)
# qc.cx(3, 2)
# qc.cx(2, 3)

print(qc.qasm())

trans_circ._layout
# circ = trans_circ.draw("mpl")
# layout = plot_circuit_layout(trans_circ, backend)
print(trans_circ.qasm())

#QuantumCircuit from open qasm
def circuit_from_openqasm(circ):
    qc = mQC(circ.num_qubits)
    layout = circ._layout.get_physical_bits()
    for key in layout:
        qc.p2v[key] = layout[key].index
        qc.v2p[layout[key].index] = key
        
    lines = circ.qasm().splitlines()
    from numpy import pi
    gates = []
    for line in lines[3:]:
        g = []
        gate, qbs = line.split(" ")
        inds = [int(qb[2]) for qb in qbs.split(",")] 
        
        if gate == "barrier":
            qc.barrier(inds)
            g.append(gate)
        
        else:
            gatename = gate[:2]
            paras = gate[2:].strip("()")
            if not len(paras) == 0:
                parastr = paras.split(",")
                paras = [eval(parai, {"pi":pi}) for parai in parastr]
                g.append(paras)    
                
            g.append(inds)
            gates.append(g)
            if gatename == "cx":
                qc.cnot(inds[0], inds[1])
            elif gatename == "u1":
                qc.rz(inds[0], paras[0])
            elif gatename == "u2":
                qc.rz(inds[0], paras[1])
                qc.ry(inds[0], pi/2)
                qc.rz(inds[0], paras[0])
            elif gatename == "u3":
                qc.rz(inds[0], paras[2])
                qc.ry(inds[0], paras[0])
                qc.rz(inds[0], paras[1])
            

    return qc

qc = circuit_from_openqasm(trans_circ)
# print(qc.gates)
qc.draw_circuit()

# #%%----------------------
# from qiskit.ignis.verification.quantum_volume import qv_circuits
# # qubit_lists = [[0,1],          # QV 4
# #                [0,1,2],        # QV 8
# #                [0,1,2,3]      # QV 16 
# #                ]

# qubit_lists = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  
#                  # QV 32
# NCIRCUITS = 1  # number of circuits to generate for each qubit list
# qv_qc_list, qv_qc_list_nomeas = qv_circuits(qubit_lists, NCIRCUITS)

# # phys = np.array([1, 6, 7, 8])
# # print(phys[qubit_lists[0]])
# device_results_list = []
# print('Running trials:\n[', end="")
# qclist = []
# for trial in range(NCIRCUITS):
#     print('#', end="")
#     print("\n")
#     qcs = qv_qc_list[trial]
#     for j in range(len(qcs)):
#         q_num = len(qubit_lists[j])
#         # inilayout = phys[qubit_lists[j]]
#         print(q_num)
#         couplings = [list(qubit_lists[j][ci:ci+2]) for ci in range(q_num-1)]
#         print(qcs[j].num_qubits)
#         print(couplings)
#         # print(inilayout)
#         t_qc = transpile(qcs[j], backend,coupling_map=couplings,optimization_level=3)
#         qclist.append(t_qc)
#     # from qiskit import assemble
#     # qobj = assemble(t_qc)
#     # result = backend.run(qobj).result()
#     # device_results_list.append(result)
# print(']')
# print('Done!')
# circ = t_qc.draw("mpl")
# layout = plot_circuit_layout(t_qc, backend)
# t_qc._layout
# print(t_qc.qasm())
# print(t_qc.depth())




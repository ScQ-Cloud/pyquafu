from .qfasm_parser import QfasmParser, QregNode
from quafu.dagcircuits.circuit_dag import node_to_gate
from quafu.dagcircuits.instruction_node import InstructionNode
from quafu.circuits import QuantumCircuit, QuantumRegister
from quafu.elements.quantum_element import Instruction


def qasm_to_circuit(qasm):
    parser = QfasmParser()
    nodes = parser.parse(qasm)

    n = 0
    gates = []
    measures = {}
    for node in nodes:
        if isinstance(node, QregNode):
            n = node.n
        if isinstance(node, InstructionNode):
            if node.name == "measure":
                for q, c in zip(node.pos.keys(), node.pos.values()):
                    measures[q] = c
            else:
                gates.append(node_to_gate(node))

    q = QuantumCircuit(n)
    q.gates = gates
    q.openqasm = qasm
    q.measures = measures
    return q


def qasm2_to_quafu_qc(qc: QuantumCircuit, openqasm: str):
    """
    Initialize pyquafu circuit from openqasm text, mainly by
    utilizing regular expressions. This function is developed
    not only to be backward compatible with pure quantum part of
    OPENQASM2.0, but also to support provide foundation of more
    powerful syntax parser.
    """
    from numpy import pi
    import re

    qc.openqasm = openqasm
    # lines = self.openqasm.strip("\n").splitlines(";")
    lines = qc.openqasm.splitlines()
    lines = [line for line in lines if line]
    qc.gates = []
    qc.measures = {}
    measured_qubits = []
    global_valid = True
    for line in lines[2:]:
        if line:
            operations_qbs = line.split(" ", 1)
            operations = operations_qbs[0]
            if operations == "qreg":
                qbs = operations_qbs[1]
                num = int(re.findall("\d+", qbs)[0])
                reg = QuantumRegister(num)
                qc.qregs.append(reg)
            elif operations == "creg":
                pass
            elif operations == "measure":
                qbs = operations_qbs[1]
                indstr = re.findall("\d+", qbs)
                inds = [int(indst) for indst in indstr]
                mb = inds[0]
                cb = inds[1]
                qc.measures[mb] = cb
                measured_qubits.append(mb)
            else:
                qbs = operations_qbs[1]
                indstr = re.findall("\d+", qbs)
                inds = [int(indst) for indst in indstr]
                valid = True
                for pos in inds:
                    if pos in measured_qubits:
                        valid = False
                        global_valid = False
                        break

                if valid:
                    if operations == "barrier":
                        qc.barrier(inds)

                    else:
                        sp_op = operations.split("(")
                        gatename = sp_op[0]
                        if gatename == "delay":
                            paras = sp_op[1].strip("()")
                            duration = int(re.findall("\d+", paras)[0])
                            unit = re.findall("[a-z]+", paras)[0]
                            qc.delay(inds[0], duration, unit)
                        elif gatename == "xy":
                            paras = sp_op[1].strip("()")
                            duration = int(re.findall("\d+", paras)[0])
                            unit = re.findall("[a-z]+", paras)[0]
                            qc.xy(min(inds), max(inds), duration, unit)
                        else:
                            if len(sp_op) > 1:
                                paras = sp_op[1].strip("()")
                                parastr = paras.split(",")
                                paras = [
                                    eval(parai, {"pi": pi}) for parai in parastr
                                ]

                            if gatename == "cx":
                                qc.cnot(inds[0], inds[1])
                            elif gatename == "cy":
                                qc.cy(inds[0], inds[1])
                            elif gatename == "cz":
                                qc.cz(inds[0], inds[1])
                            elif gatename == "cp":
                                qc.cp(inds[0], inds[1], paras[0])
                            elif gatename == "swap":
                                qc.swap(inds[0], inds[1])
                            elif gatename == "rx":
                                qc.rx(inds[0], paras[0])
                            elif gatename == "ry":
                                qc.ry(inds[0], paras[0])
                            elif gatename == "rz":
                                qc.rz(inds[0], paras[0])
                            elif gatename == "p":
                                qc.p(inds[0], paras[0])
                            elif gatename == "x":
                                qc.x(inds[0])
                            elif gatename == "y":
                                qc.y(inds[0])
                            elif gatename == "z":
                                qc.z(inds[0])
                            elif gatename == "h":
                                qc.h(inds[0])
                            elif gatename == "id":
                                qc.id(inds[0])
                            elif gatename == "s":
                                qc.s(inds[0])
                            elif gatename == "sdg":
                                qc.sdg(inds[0])
                            elif gatename == "t":
                                qc.t(inds[0])
                            elif gatename == "tdg":
                                qc.tdg(inds[0])
                            elif gatename == "sx":
                                qc.sx(inds[0])
                            elif gatename == "ccx":
                                qc.toffoli(inds[0], inds[1], inds[2])
                            elif gatename == "cswap":
                                qc.fredkin(inds[0], inds[1], inds[2])
                            elif gatename == "u1":
                                qc.rz(inds[0], paras[0])
                            elif gatename == "u2":
                                qc.rz(inds[0], paras[1])
                                qc.ry(inds[0], pi / 2)
                                qc.rz(inds[0], paras[0])
                            elif gatename == "u3":
                                qc.rz(inds[0], paras[2])
                                qc.ry(inds[0], paras[0])
                                qc.rz(inds[0], paras[1])
                            elif gatename == "rxx":
                                qc.rxx(inds[0], inds[1], paras[0])
                            elif gatename == "ryy":
                                qc.ryy(inds[0], inds[1], paras[0])
                            elif gatename == "rzz":
                                qc.rzz(inds[0], inds[1], paras[0])
                            else:
                                print(
                                    "Warning: Operations %s may be not supported by QuantumCircuit class currently."
                                    % gatename
                                )

    if not qc.measures:
        qc.measures = dict(zip(range(qc.num), range(qc.num)))
    if not global_valid:
        print(
            "Warning: All operations after measurement will be removed for executing on experiment"
        )


if __name__ == '__main__':
    import re

    pattern = r"[a-z]"

    text = "Hello, world! This is a test."

    matches = re.findall(pattern, text)

    print(matches)

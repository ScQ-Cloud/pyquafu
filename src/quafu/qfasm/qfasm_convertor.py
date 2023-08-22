from .qfasm_parser import QfasmParser, QregNode
from quafu.dagcircuits.circuit_dag import node_to_gate
from quafu.dagcircuits.instruction_node import InstructionNode
from quafu.circuits import QuantumCircuit, QuantumRegister


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

    lines = qc.openqasm.splitlines()
    lines = [line for line in lines if line]
    lines = [line for line in lines if not line.startswith("//")]  # annotations

    # init qc
    qc.gates = []
    qc.measures = {}
    measured_qubits = []
    global_valid = True

    # proceed line by line
    for line in lines[2:]:
        if line:
            operations_qbs = line.split(" ", 1)
            operations = operations_qbs[0]
            if operations == "qreg":
                qbs = operations_qbs[1]
                num = int(re.findall(r"\d+", qbs)[0])
                reg = QuantumRegister(num)
                qc.qregs.append(reg)
            elif operations == "creg":
                pass
            elif operations == "measure":
                qbs = operations_qbs[1]
                indstr = re.findall(r"\d+", qbs)
                inds = [int(indst) for indst in indstr]
                mb = inds[0]
                cb = inds[1]
                qc.measures[mb] = cb
                measured_qubits.append(mb)
            else:
                # apply some kind of instruction
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
                        sp_op = operations.split("(")  # parameters
                        gatename = sp_op[0].lower()
                        if gatename == "delay":
                            paras = sp_op[1].strip("()")
                            duration = int(re.findall(r"\d+", paras)[0])
                            unit = re.findall("[a-z]+", paras)[0]
                            qc.delay(inds[0], duration, unit)
                        elif gatename == "xy":
                            paras = sp_op[1].strip("()")
                            duration = int(re.findall(r"\d+", paras)[0])
                            unit = re.findall("[a-z]+", paras)[0]
                            qc.xy(min(inds), max(inds), duration, unit)
                        else:
                            if len(sp_op) > 1:
                                paras = sp_op[1].strip("()")
                                parastr = paras.split(",")
                                paras = [
                                    eval(parai, {"pi": pi}) for parai in parastr
                                ]

                            if gatename in ["cx", "cy", "cz", "swap"]:
                                funcs = {"cx": qc.cnot, "cy": qc.cy, "cz": qc.cz, "swap": qc.swap}
                                funcs[gatename](inds[0], inds[1])
                            elif gatename == "cp":
                                qc.cp(inds[0], inds[1], paras[0])
                            elif gatename in ["rx", "ry", "rz", "p"]:
                                funcs = {"rx": qc.rx, "ry": qc.ry, "rz": qc.rz, "p": qc.p}
                                funcs[gatename](inds[0], paras[0])
                            elif gatename in ["x", "y", "z", "h", "id", "s", "sdg", "t", "tdg", "sx"]:
                                func = {"x": qc.x, "y": qc.y, "z": qc.z, "h": qc.h, "id": qc.id,
                                        "s": qc.s, "sdg": qc.sdg, "t": qc.t, "tdg": qc.tdg, "sx": qc.sx}
                                func[gatename](inds[0])
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
                            elif gatename in ["rxx", "ryy", "rzz"]:
                                funcs = {"rxx": qc.rxx, "ryy": qc.ryy, "rzz": qc.rzz}
                                funcs[gatename](inds[0], inds[1], paras[0])
                            else:
                                print(
                                    "Warning: Operations %s may be not supported by QuantumCircuit class currently."
                                    % gatename
                                )

    if not qc.measures:
        qc.measures = dict(zip(range(qc.num), range(qc.num)))
    if not global_valid:
        import warnings
        warnings.warn("All operations after measurement will be removed for executing on experiment")

    return qc

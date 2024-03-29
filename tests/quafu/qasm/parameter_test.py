import math

from quafu import QuantumCircuit
from quafu.qfasm.qfasm_convertor import qasm_to_quafu
from quafu.elements import ParameterExpression, Parameter

class TestParser:
    """
    Test for PLY parser
    """

    def compare_cir(self, qc1: QuantumCircuit, qc2: QuantumCircuit):
        # compare reg and compare gates
        assert len(qc1.qregs) == len(qc2.qregs)
        for i in range(len(qc1.qregs)):
            reg1 = qc1.qregs[i]
            reg2 = qc2.qregs[i]
            assert len(reg1.qubits) == len(reg2.qubits)
        assert len(qc1.gates) == len(qc2.gates)
        assert len(qc1.variables) == len(qc2.variables)
        for i in range(len(qc1.variables)):
            self.compare_parameter(qc1.variables[i], qc2.variables[i])
        for i in range(len(qc1.gates)):
            gate1 = qc1.gates[i]
            gate2 = qc2.gates[i]
            assert gate1.name == gate2.name
            if hasattr(gate1, "pos"):
                assert gate1.pos == gate2.pos
            if hasattr(gate1, "paras"):
                assert len(gate1.paras) == len(gate2.paras)
                for j in range(len(gate1.paras)):
                    self.compare_parameter(gate1.paras[j], gate2.paras[j])

    def compare_parameter(self, param1, param2):
        if isinstance(param1, ParameterExpression) or isinstance(param2, ParameterExpression):
            assert isinstance(param2, ParameterExpression)
            assert isinstance(param1, ParameterExpression)
            assert param1.latex == param2.latex
            assert param1.value == param2.value
            assert len(param1.funcs) == len(param2.funcs)
            assert len(param1.operands) == len(param2.operands)
            if not param1.latex:
                self.compare_parameter(param1.pivot, param2.pivot)
            for i in range(len(param1.funcs)):
                assert param1.funcs[i] == param2.funcs[i]
            for i in range(len(param1.operands)):
                self.compare_parameter(param1.operands[i], param2.operands[i])
        else:
            assert param1 == param2
    # ----------------------------------------
    #   test for parameter
    # ----------------------------------------
    def test_parameter_plain(self):
        qasm = """
                theta1 = 1.0; theta2 = 2.0; qreg q[2]; rx(theta1) q[0]; rx(theta2) q[1];
                """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RX"
        assert cir.gates[1].name == "RX"
        self.compare_parameter(cir.variables[0], Parameter("theta1", 1.0))
        self.compare_parameter(cir.variables[1], Parameter("theta2", 2.0))
        self.compare_parameter(cir.gates[0].paras[0], Parameter("theta1", 1.0))
        self.compare_parameter(cir.gates[1].paras[0], Parameter("theta2", 2.0))

    def test_parameter_func(self):
        qasm = """
               theta1 = 1.0; theta2 = 2.0; 
               gate test(rz, ry) a {
                    rz(rz) a;
                    ry(ry) a;
                }
                qreg q[1];
                test(theta1, theta2) q[0];
            """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RZ"
        assert cir.gates[1].name == "RY"
        self.compare_parameter(cir.variables[0], Parameter("theta1", 1.0))
        self.compare_parameter(cir.variables[1], Parameter("theta2", 2.0))
        self.compare_parameter(cir.gates[0].paras[0], Parameter("theta1", 1.0))
        self.compare_parameter(cir.gates[1].paras[0], Parameter("theta2", 2.0))

    def test_parameter_func_mix(self):
        qasm = """
                theta = 1.0;
                theta1 = 3.0;
                gate test(rz, theta1) a {
                    rz(rz) a;
                    ry(theta1) a;
                }
                qreg q[1];
                test(theta, 2.0) q[0];
            """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RZ"
        assert cir.gates[1].name == "RY"
        assert len(cir.variables) == 1
        assert cir.gates[1].paras[0] == 2.0
        self.compare_parameter(cir.variables[0], Parameter("theta", 1.0))
        self.compare_parameter(cir.gates[0].paras[0], Parameter("theta", 1.0))

    def test_parameter_func_mix2(self):
        qasm = """
                theta = 1.0;
                theta1 = 3.0;
                gate test(rz, theta1) a {
                    rz(rz) a;
                    ry(theta) a;
                }
                qreg q[1];
                test(theta, 2.0) q[0];
            """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RZ"
        assert cir.gates[1].name == "RY"
        assert len(cir.variables) == 1
        self.compare_parameter(cir.variables[0], Parameter("theta", 1.0))
        self.compare_parameter(cir.gates[0].paras[0], Parameter("theta", 1.0))
        self.compare_parameter(cir.gates[1].paras[0], Parameter("theta", 1.0))

    def test_parameter_expression(self):
        qasm = """
                theta1 = 1.0; theta2 = 2.0; 
                qreg q[2]; rx(theta1+theta2) q[0]; rx(theta1+theta1*theta2) q[1];
            """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RX"
        assert cir.gates[1].name == "RX"
        theta1 = Parameter("theta1", 1.0)
        theta2 = Parameter("theta2", 2.0)
        self.compare_parameter(cir.variables[0], theta1)
        self.compare_parameter(cir.variables[1], theta2)
        self.compare_parameter(cir.gates[0].paras[0], theta1+theta2)
        self.compare_parameter(cir.gates[1].paras[0], theta1+theta1*theta2)
        qasm = """
                theta1 = 1.0; theta2 = 2.0;
                theta3 = 3.0; theta4 = 4.0; 
                qreg q[2]; 
                rx(theta1+theta2-theta3*theta4^2) q[0]; 
                rx(sin(theta1*2+theta2)) q[1];
            """
        cir = qasm_to_quafu(openqasm=qasm)
        assert cir.gates[0].name == "RX"
        assert cir.gates[1].name == "RX"
        theta1 = Parameter("theta1", 1.0)
        theta2 = Parameter("theta2", 2.0)
        theta3 = Parameter("theta3", 3.0)
        theta4 = Parameter("theta4", 4.0)
        assert len(cir.variables) == 4
        theta = theta1 + theta2 - theta3 * theta4 ** 2
        self.compare_parameter(cir.variables[0], theta1)
        self.compare_parameter(cir.variables[1], theta2)
        self.compare_parameter(cir.variables[2], theta3)
        self.compare_parameter(cir.variables[3], theta4)
        self.compare_parameter(cir.gates[0].paras[0], theta)
        theta = (theta1*2+theta2).sin()
        self.compare_parameter(cir.gates[1].paras[0], theta)


    def test_parameter_to_from(self):
        qc = QuantumCircuit(4)
        theta1 = Parameter("theta1", 1.0)
        theta2 = Parameter("theta2", 2.0)
        qc.rx(0, theta1)
        qc.rx(0, theta2)
        qc2 = QuantumCircuit(4)
        qc2.from_openqasm(qc.to_openqasm(with_para=True))
        self.compare_cir(qc,qc2)
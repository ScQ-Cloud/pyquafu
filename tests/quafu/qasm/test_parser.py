import math
import os
import pathlib
import random
import sys
import tempfile

import pytest

from quafu.circuits import QuantumCircuit
from unittest.mock import patch
from quafu.qfasm.exceptions import LexerError, ParserError
from quafu.qfasm.qfasm_convertor import qasm_to_quafu

class TestLexer:
    """
    Test for PLY parser
    """
    def compare_cir(self, qc1:QuantumCircuit, qc2:QuantumCircuit):
        # compare reg and compare gates
        assert(len(qc1.qregs) == len(qc2.qregs))
        for i in range(len(qc1.qregs)):
            reg1 = qc1.qregs[i]
            reg2 = qc2.qregs[i]
            assert(len(reg1.qubits) == len(reg2.qubits))
        assert(len(qc1.gates) == len(qc2.gates))
        for i in range(len(qc1.gates)):
            gate1 = qc1.gates[i]
            gate2 = qc2.gates[i]
            assert(gate1.name==gate2.name)
            if hasattr(gate1, 'pos'):
                assert(gate1.pos == gate2.pos)
            if hasattr(gate1,'paras'):
                assert(gate1.paras == gate2.paras)
        assert qc1.to_openqasm() == qc2.to_openqasm()

    #----------------------------------------
    #   test for lexer
    #----------------------------------------
    def test_id_cannot_start_with_capital(self):
        with pytest.raises(LexerError, match=r"Illegal character .*") as e:
            token = "qreg Qcav[1];"
            qasm_to_quafu(token)

    def test_unexpected_linebreak(self):
        with pytest.raises(LexerError) as e:
            token = 'include "qe\nlib1.inc";'
            qasm_to_quafu(token)
    
    def test_include_miss_sigd(self):
        with pytest.raises(LexerError, match=r'Expecting ";" for INCLUDE at line.*') as e:
            token = 'include "qelib1.inc" qreg q[1];'
            qasm_to_quafu(token)

    @pytest.mark.parametrize('filename', ['qelib1.inc', '"qelib1.inc', 'qelib1.inc"'])
    def test_filename_miss_quote(self, filename):
        with pytest.raises(LexerError, match=r"Invalid include: need a quoted string as filename.") as e:
            token = f'include {filename};'
            qasm_to_quafu(token)
    
    def test_include_cannot_find_file(self):
        with pytest.raises(LexerError, match=r"Include file .* cannot be found, .*") as e:
            token = f'include "qe.inc";'
            qasm_to_quafu(token)

    def test_single_equals_error(self):
        with pytest.raises(LexerError, match=r"Illegal character =.*") as e:
            qasm = f'if (a=2) U(0,0,0)q[0];'
            qasm_to_quafu(qasm)

    def test_invalid_token(self):
        with pytest.raises(LexerError, match=r"Illegal character.*") as e:
            token = f'!'
            qasm_to_quafu(token)

    def test_single_quote_filename(self):
        with pytest.raises(LexerError, match=r"Illegal character '.*") as e:
            token = f"include 'qe.inc';"
            qasm_to_quafu(token)

    #----------------------------------------
    #   test for expression
    #----------------------------------------
    def test_exp_unary(self):
        num1 = random.random()
        qasm = f"qreg q[1]; U(0, 0, {num1})q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == num1)
    
    def test_exp_unary_symbolic(self):
        qasm = """
        gate test(a,b,c) q{
            U(a+b,-(a-c), a+c-b) q;
        }
        qreg q[1]; test(0.5, 1.0, 2.0)q[0];
        """
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == 0.5+2.0-1.0)
        assert(cir.gates[1].paras == 0.5+1.0)
        assert(cir.gates[2].paras == -(0.5-2.0))

    def test_exp_add(self):
        num1 = random.random()
        qasm = f"qreg q[1]; U(0, 0, {num1} + 1 + 1)q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == num1+1+1)
    
    def test_exp_sub(self):
        num1 = random.random()
        qasm = f"qreg q[1]; U(0, 0, 3 - {num1})q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == 3-num1)

    def test_exp_mul(self):
        num1 = random.random()
        qasm = f"qreg q[1]; U(0, 0, 2 * {num1})q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == 2*num1)
    
    def test_exp_div(self):
        num1 = random.random()
        qasm = f"qreg q[1]; U(0, 0, {num1}/2)q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == num1/2)
    
    def test_exp_power(self):
        qasm = f"qreg q[1]; U(0, 0, 2^3)q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == 2**3)

    @pytest.mark.parametrize(['symbol','op'], [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b),
        ("^", lambda a, b: a**b),
    ])
    def test_exp_binary(self, symbol, op):
        num1 = random.random()
        num2 = random.random()
        qasm = f"qreg q[1]; U(0, 0, {num1}{symbol}{num2})q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == op(num1,num2))

    @pytest.mark.parametrize(['symbol','op'], [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b),
        ("^", lambda a, b: a**b),
    ])
    def test_exp_binary_symbol(self, symbol, op):
        num1 = random.random()
        num2 = random.random()
        qasm = f"""
        gate test(a,b,c) q {{
            U(0,0,a{symbol}(b{symbol}c)) q;
        }}
        qreg q[1]; 
        test({num2}, {num1}, {num2})q[0];
        """
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == op(num2,op(num1,num2)))

    def test_exp_mix(self):
        qasm = f"qreg q[1]; U(2+1*2-3, 1+3-2, 2/2*2)q[0]; U(3*(2+1),3-(2-1)*2,(2-1)+1)q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == 2/2*2)
        assert(cir.gates[1].name == 'RY')
        assert(cir.gates[1].paras == 2+1*2-3)
        assert(cir.gates[2].name == 'RZ')
        assert(cir.gates[2].paras == 1+3-2)
        assert(cir.gates[3].name == 'RZ')
        assert(cir.gates[3].paras == (2-1)+1)
        assert(cir.gates[4].name == 'RY')
        assert(cir.gates[4].paras == 3*(2+1))
        assert(cir.gates[5].name == 'RZ')
        assert(cir.gates[5].paras == 3-(2-1)*2)
    
    def test_exp_par(self):
        qasm = f"qreg q[1]; U((1-2)+(3-2), -(2*2-2), 2*(2-3))q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == 2*(2-3))
        assert(cir.gates[1].name == 'RY')
        assert(cir.gates[1].paras == (1-2)+(3-2))
        assert(cir.gates[2].name == 'RZ')
        assert(cir.gates[2].paras == -(2*2-2))
    
    @pytest.mark.parametrize(['func','mathop'], [
        ("cos", math.cos),
        ("exp", math.exp),
        ("ln", math.log),
        ("sin", math.sin),
        ("sqrt", math.sqrt),
        ("tan", math.tan),
    ])
    def test_exp_func(self, func, mathop):
        qasm = f'qreg q[1]; U({func}(0.5),{func}(1.0),{func}(pi)) q[0];'
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == mathop(math.pi))
        assert(cir.gates[1].paras == mathop(0.5))
        assert(cir.gates[2].paras == mathop(1.0))

    @pytest.mark.parametrize(['func','mathop'], [
        ("cos", math.cos),
        ("exp", math.exp),
        ("ln", math.log),
        ("sin", math.sin),
        ("sqrt", math.sqrt),
        ("tan", math.tan),
    ])
    def test_exp_func_symbol(self, func, mathop):
        num1 = random.random()
        num2 = random.random()
        num3 = random.random()
        qasm = f"""
        gate test(a,b,c) q{{
            U({func}(a),{func}(b),{func}(c)) q;
        }}
        qreg q[1]; 
        test({num1},{num2},{num3}) q[0];
        """
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == mathop(num3))
        assert(cir.gates[1].paras == mathop(num1))
        assert(cir.gates[2].paras == mathop(num2))

    def test_exp_precedence(self):
        num1 = random.random()
        num2 = random.random()
        num3 = random.random()
        expr = f"{num1} + 1.5 * -{num3}  ^ 2 - {num2} / 0.5"
        # expr1 = f"{num1} * -{num2} ^ 2"
        # expected1 = num1* (-num2)**2
        expected = num1 + 1.5 * (-num3)**2 - num2 / 0.5
        qasm = f"qreg q[1]; U( 0, 0, {expr}) q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == expected)

    def test_exp_sub_left(self):
        qasm = f"qreg q[1]; U( 0 , 0 , 2.0-1.0-1.0 ) q[0];"
        print(qasm)
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == 0)
        assert(cir.gates[1].paras == 0)
        assert(cir.gates[2].paras == 0)
    
    def test_exp_div_left(self):
        qasm = f"qreg q[1]; U( 0 , 0 , 2.0/2.0/2.0 ) q[0];"
        print(qasm)
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == 0.5)
        assert(cir.gates[1].paras == 0)
        assert(cir.gates[2].paras == 0)

    def test_exp_pow_right(self):
        qasm = f"qreg q[1]; U( 0 , 0 , 2.0^3.0^2.0 ) q[0];"
        print(qasm)
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].paras == 512.0)
        assert(cir.gates[1].paras == 0)
        assert(cir.gates[2].paras == 0)

    def test_exp_div_zero(self):
        with pytest.raises(ParserError, match=r"Divided by 0 at line.*") as e:
            qasm = f"qreg q[1]; U( 0 , 0 , 1.0/0 ) q[0];"
            qasm_to_quafu(openqasm=qasm)
        with pytest.raises(ParserError, match=r"Divided by 0 at line.*") as e:
            qasm = f"qreg q[1]; U( 0 , 0 , 1.0/(1.0-1.0) ) q[0];"
            qasm_to_quafu(openqasm=qasm)

    def test_exp_ln_neg(self):
        "It's send to numpy.log, so will get nan"
        pass
    
    def test_exp_sqrt_neg(self):
        "It's send to numpy.sqrt, so will get nan"
        pass
    
    @pytest.mark.parametrize(['symbol','op'], [
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b),
        ("^", lambda a, b: a**b),
    ])
    def test_exp_nonunary_operators_lack(self, symbol, op):
        qasm = f"qreg q[1]; U( 0 , 0 , {symbol}1.0 ) q[0];"
        with pytest.raises(ParserError, match=r"Expected an ID, received.*") as e:
            qasm_to_quafu(openqasm=qasm) 

    @pytest.mark.parametrize(['symbol','op'], [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b),
        ("^", lambda a, b: a**b),
    ])
    def test_exp_missing_binary_operand(self, symbol, op):
        qasm = f"qreg q[1]; U( 0 , 0 , 1.0{symbol} ) q[0];"
        with pytest.raises(ParserError, match=r"Expected an ID, received.*") as e:
            qasm_to_quafu(openqasm=qasm) 
    
    def test_exp_missing_op(self):
        qasm = f"qreg q[1]; U( 0 , 0 , 1.0 2.0 ) q[0];"
        with pytest.raises(ParserError, match=r"Missing '\)' after '\('.*") as e:
            qasm_to_quafu(openqasm=qasm) 

    def test_exp_missing_premature_right_pare(self):
        qasm = f"qreg q[1]; U( 0 , 0 , sin() ) q[0];"
        with pytest.raises(ParserError, match=r"Expected an ID, received.*") as e:
            qasm_to_quafu(openqasm=qasm) 

    #----------------------------------------
    #   test for parser
    #----------------------------------------
    def test_multi_format(self):
        qasm = """
        OPENQASM
        2.0
        ;
        gate    // a comment here
        test(   // to split a gate declaration
        theta
        )       // But it's
        q       // still a 
        { h     // void
q;              // gate!
u2(
theta ,
        1) q
        ;}
        qreg    // a quantum reg
        q
        [2];
        test(0.1) q[0]; cx q[0],
        q[1];
        creg c[2]; measure q->c;"""

        cir = qasm_to_quafu(openqasm=qasm)
        expected_cir = QuantumCircuit(2)
        expected_cir.h(0)
        expected_cir.rz(0,1)
        expected_cir.ry(0,math.pi/2)
        expected_cir.rz(0,0.1)
        expected_cir.cx(0,1)
        expected_cir.measure([0,1])
        self.compare_cir(cir, expected_cir)

    @pytest.mark.parametrize('num', ['0.11','2.2', '0.44'])
    def  test_float(self, num):
        qasm = f"qreg q[1]; U(0, 0, {num})q[0];"
        cir = qasm_to_quafu(openqasm=qasm)
        assert(cir.gates[0].name == 'RZ')
        assert(cir.gates[0].paras == float(num))
    
    def test_id_cannot_start_with_num(self):
        with pytest.raises(ParserError, match=r"Expected an ID, received .*") as e:
            token = "qreg 0cav[1];"
            qasm_to_quafu(token)
    
    def test_openqasm_float(self):
        with pytest.raises(ParserError, match=r"Expected FLOAT after OPENQASM, received .*") as e:
            token = "OPENQASM 3;"
            qasm_to_quafu(token)

    def test_openqasm_version(self):
        with pytest.raises(ParserError, match=r"Only support OPENQASM 2.0 version.*") as e:
            token = "OPENQASM 3.0;"
            qasm_to_quafu(token)

    def test_openqasm_miss_sigd(self):
        with pytest.raises(ParserError, match=r"Missing ';' at end of OPENQASM statement.*") as e:
            token = "OPENQASM 2.0 qreg q[3];"
            qasm_to_quafu(token)
    
    def test_unexpected_end_of_file(self):
        with pytest.raises(ParserError, match=r"Error at end of file") as e:
            token = "OPENQASM 2.0"
            qasm_to_quafu(token)

    def test_allow_empty(self):
        cir = qasm_to_quafu("")
        self.compare_cir(cir, QuantumCircuit(0))

    def test_comment(self):
        qasm = """
            // It's a comment
            OPENQASM 2.0;
            qreg q[2];
        """
        cir = qasm_to_quafu(qasm)
        self.compare_cir(cir, QuantumCircuit(2))
    
    def test_register(self):
        qasm = """
            OPENQASM 2.0;
            qreg q[2];
            qreg c[1];
        """
        cir = qasm_to_quafu(qasm)
        self.compare_cir(cir, QuantumCircuit(3))
    
    def test_creg(self):
        qasm = """
            OPENQASM 2.0;
            creg q[2];
            creg c[1];
        """
        cir = qasm_to_quafu(qasm)
        self.compare_cir(cir, QuantumCircuit(0))
    
    def test_registers_after_gate(self):
        qasm = "qreg a[2]; CX a[0], a[1]; qreg b[2]; CX b[0], b[1];"
        cir = qasm_to_quafu(qasm)
        # print(cir.gates)
        qc = QuantumCircuit(4)
        qc.cx(0,1)
        qc.cx(2,3)
        self.compare_cir(cir,qc)
    
    def test_builtin_single(self):
        qasm = """
            qreg q[2];
            U(0, 0, 0) q[0];
            CX q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.rz(0,0)
        qc.ry(0,0)
        qc.rz(0,0)
        qc.cx(0,1)
        self.compare_cir(cir,qc)

    def test_builtin_broadcast(self):
        qasm = """
            qreg q[2];
            U(0, 0, 0) q;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.rz(0,0)
        qc.ry(0,0)
        qc.rz(0,0)
        qc.rz(1,0)
        qc.ry(1,0)
        qc.rz(1,0)
        self.compare_cir(cir,qc)

    def test_builtin_broadcast_2q(self):
        qasm = """
            qreg q[2];
            qreg r[2];
            CX q,r;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(4)
        qc.cx(0,2)
        qc.cx(1,3)
        self.compare_cir(cir,qc)

    def test_call_defined_gate(self):
        qasm = """
            gate test a,b {
                CX a,b;
            }
            qreg q[2];
            test q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.cx(0,1)
        self.compare_cir(cir,qc)

    def test_parameterless_gates(self):
        qasm = """
            qreg q[2];
            CX q[0], q[1];
            CX() q[1], q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.cx(0,1)
        qc.cx(1,0)
        self.compare_cir(cir,qc)
    
    def test_include_in_definition(self):
        qasm = """
            include "qelib1.inc";
            qreg q[2];
            cx q[0], q[1];
            cx() q[1], q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.cx(0,1)
        qc.cx(1,0)
        self.compare_cir(cir,qc)

    def test_previous_defined_gate(self):
        qasm = """
            include "qelib1.inc";
            gate bell a, b {
                h a;
                cx a, b;
            }
            gate second_bell a, b {
                bell b, a;
            }
            qreg q[2];
            second_bell q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cx(1,0)
        self.compare_cir(cir,qc)
    
    def test_qubitname_lookup_differently_to_gate(self):
        qasm = """
            gate bell h, cx {
                h h;
                cx h, cx;
            }
            qreg q[2];
            bell q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0,1)
        self.compare_cir(cir,qc)

    def test_paramname_lookup_differently_to_gates(self):
        qasm = """
            gate test(rz, ry) a {
                rz(rz) a;
                ry(ry) a;
            }
            qreg q[1];
            test(0.5, 1.0) q[0];
        """  
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        qc.rz(0, 0.5)
        qc.ry(0, 1.0)
        self.compare_cir(cir,qc)

    def test_unused_parameter(self):
        qasm = """
            gate test(rz, ry) a {
                rz(1.0) a;
                ry(2.0) a;
            }
            qreg q[1];
            test(0.5, 1.0) q[0];
        """  
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        qc.rz(0, 1.0)
        qc.ry(0, 2.0)
        self.compare_cir(cir,qc)

    def test_qubit_barrier_in_definition(self):
        qasm = """
            gate test a, b {
                barrier a;
                barrier b;
                barrier a, b;
            }
            qreg q[2];
            test q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.barrier([0])
        qc.barrier([1])
        qc.barrier([0])
        qc.barrier([1])
        self.compare_cir(cir,qc)
    
    def test_double_call_gate(self):
        qasm = """
            gate test(x, y) a {
                rz(x) a;
                ry(y) a;
            }
            qreg q[2];
            test(0.5, 1.0) q[0];
            test(1.0, 0.5) q[1];
        """  
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.rz(0, 0.5)
        qc.ry(0, 1.0)
        qc.rz(1, 1.0)
        qc.ry(1, 0.5)
        self.compare_cir(cir,qc)

    def test_muti_register(self):
        qasm = """
            qreg q[2];
            qreg r[2];
            qreg m[3];
        """  
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(7)
        self.compare_cir(cir,qc)

    def test_single_measure(self):
        qasm = """
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        qc.measure([0])
        self.compare_cir(cir,qc)

    def test_muti_measure(self):
        qasm = """
            qreg q[2];
            creg c[2];
            measure q -> c;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.measure([0,1])
        self.compare_cir(cir,qc)

    def test_single_reset(self):
        qasm = """
            qreg q[1];
            reset q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        qc.reset([0])
        self.compare_cir(cir,qc)

    def test_muti_reset(self):
        qasm = """
            qreg q[2];
            reset q;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.reset([0,1])
        self.compare_cir(cir,qc)
    
    def test_include_all_gate(self):
        qasm = """
            include "qelib1.inc";
            qreg q[3];
            u3(0.5, 0.25, 0.125) q[0];
            u2(0.5, 0.25) q[0];
            u1(0.5) q[0];
            cx q[0], q[1];
            id q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            rx(0.5) q[0];
            ry(0.5) q[0];
            rz(0.5) q[0];
            cz q[0], q[1];
            cy q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(3)
        qc.rz(0,0.125)
        qc.ry(0,0.5)
        qc.rz(0,0.25)
        qc.rz(0,0.25)
        qc.ry(0,math.pi/2)
        qc.rz(0,0.5)
        qc.rz(0,0.5)
        qc.ry(0,0)
        qc.rz(0,0)
        qc.cx(0,1)
        qc.id(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.rx(0,0.5)
        qc.ry(0,0.5)
        qc.rz(0,0.5)
        qc.cz(0,1)
        qc.cy(0,1)
        self.compare_cir(cir,qc)

    def test_include_define_version(self):
        include = """
            OPENQASM 2.0;
            qreg q[2];
        """
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        
        with open(tmp_dir / "include.qasm", "w") as fp:
            fp.write(include)
        filepath = tmp_dir / "include.qasm"
        qasm = f'OPENQASM 2.0;include "{filepath}";'
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        self.compare_cir(cir, qc)

    def test_nested_include(self):
        inner = "qreg c[2];"
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        with open(tmp_dir / "inner.qasm", "w") as fp:
            fp.write(inner)
        filepath = tmp_dir / "inner.qasm"    
        outer = f"""
            qreg q[2];
            include "{filepath}";
        """
        with open(tmp_dir / "outer.qasm", "w") as fp:
            fp.write(outer)
        filepath = tmp_dir / "outer.qasm"
        qasm = f"""
            OPENQASM 2.0;
            include "{filepath}";
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(4)
        self.compare_cir(cir, qc)

    def test_include_from_current_directory(self):
        include = """
            qreg q[2];
        """
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        with open(tmp_dir / "include.qasm", "w") as fp:
            fp.write(include)
        qasm = """
            OPENQASM 2.0;
            include "include.qasm";
        """
        prevdir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            cir = qasm_to_quafu(qasm)
            qc = QuantumCircuit(2)
            self.compare_cir(cir, qc)
        finally:
            os.chdir(prevdir) 

    def test_override_cx(self):
        qasm = """
            qreg q[2];
            CX q[0], q[1];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.cx(0,1)
        self.compare_cir(cir,qc)

    def test_override_u3(self):
        qasm = """
            qreg q[2];
            U(0.1,0.2,0.3) q[0] ;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        qc.rz(0,0.3)
        qc.ry(0,0.1)
        qc.rz(0,0.2)
        self.compare_cir(cir,qc)

    def test_gate_without_body(self):
        qasm = """
            qreg q[2];
            gate test( ) q { }
            test q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        self.compare_cir(cir,qc)

    def test_gate_params_without_body(self):
        qasm = """
            qreg q[2];
            gate test(x,y) q { }
            test(0.1,0.2) q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(2)
        self.compare_cir(cir,qc)

    def test_qiskit_mathfunc(self):
        qasm = """
            include "qelib1.inc";
            qreg q[1];
            rx(asin(0.3)) q[0];
            ry(acos(0.3)) q[0];
            rz(atan(0.3)) q[0];
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        qc.rx(0,math.asin(0.3))
        qc.ry(0,math.acos(0.3))
        qc.rz(0,math.atan(0.3))
        self.compare_cir(cir,qc)

    def test_empty_statement(self):
        qasm="""
        include "qelib1.inc";
        qreg q[1];
        ;
        """
        cir = qasm_to_quafu(qasm)
        qc = QuantumCircuit(1)
        self.compare_cir(cir,qc)


    # parser error
    def test_registers_size_positive(self):
        qasm = "qreg a[0]; CX a[0], a[1]; qreg b[2]; CX b[0], b[1];"    
        with pytest.raises(ParserError, match=r"QREG size must be positive at line") :
            qasm_to_quafu(qasm)

    
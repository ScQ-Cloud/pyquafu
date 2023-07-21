from quafu.circuits.quantum_circuit import QuantumCircuit
import matplotlib.pyplot as plt

"""
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
"""

n = 10
qc = QuantumCircuit(n)


def majority(a, b, c):
    """
    gate majority a,b,c
    {
      cx c,b;
      cx c,a;
      ccx a,b,c;
    }
    """
    qc.cnot(c, b)
    qc.cnot(a, b)
    qc.mcx([a, b], c)


def unmaj(a, b, c):
    """
    gate unmaj a,b,c
    {
      ccx a,b,c;
      cx c,a;
      cx a,b;
    }
    """
    qc.mcx([a, b], c)
    qc.cnot(c, a)
    qc.cnot(a, b)


def qreg(_i, name):
    """
    qreg cin[1];
    qreg a[4];
    qreg b[4];
    qreg cout[1];
    """
    if name == 'cin':
        return _i
    elif name == 'a':
        return _i + 1
    elif name == 'b':
        return _i + 5
    elif name == 'cout':
        return _i + 9
    else:
        raise ValueError('Unknown qreg name: {}'.format(name))


def creg(_i, name):
    """
    creg ans[5];
    """
    if name == 'ans':
        return _i
    else:
        raise ValueError('Unknown creg name: {}'.format(name))


"""
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
"""
qc.x(qreg(0, 'a'))
for i in range(4):
    qc.x(qreg(i, 'b'))

"""
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
"""
majority(qreg(0, 'cin'), qreg(0, 'b'), qreg(0, 'a'))
majority(qreg(0, 'a'), qreg(1, 'b'), qreg(1, 'a'))
for i in range(1, 4):
    majority(qreg(i-1, 'a'), qreg(i, 'b'), qreg(i, 'a'))
qc.cnot(qreg(3, 'a'), qreg(0, 'cout'))
unmaj(qreg(2, 'a'), qreg(3, 'b'), qreg(3, 'a'))
unmaj(qreg(1, 'a'), qreg(2, 'b'), qreg(2, 'a'))
unmaj(qreg(0, 'a'), qreg(1, 'b'), qreg(1, 'a'))

"""
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
"""
measure_pos = [qreg(i, 'b') for i in range(4)] + [qreg(0, 'cout')]
measure_cbits = [creg(i, 'ans') for i in range(5)]
qc.measure(measure_pos, cbits=measure_cbits)
# qc.draw_circuit()
# print(qc.to_openqasm())

init_labels = dict.fromkeys(range(n))
init_labels[0] = 'cin'
for i in range(4):
    init_labels[i+1] = f'a_{i}'
for i in range(5):
    init_labels[i+5] = f'b_{i}'

end_labels = {i+5: f'ans_{i}' for i in range(5)}
qc.plot_circuit(title='Quantum ripple-carry adder',
                init_labels=init_labels,
                end_labels=end_labels,
                )
plt.show()

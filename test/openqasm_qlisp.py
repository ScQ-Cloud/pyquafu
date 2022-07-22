#%%---------
test = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[0];
h q[1];
measure q[0] -> c[1];
cx q[0],q[2];
cx q[1],q[2];
x q[2];
ry(pi) q[3];
u1(pi/2) q[1];
measure q[2] -> c[3];
x q[2];
cx q[3],q[4];
"""

def cut_openqasm(openqasm):
    import re
    lines = openqasm.splitlines()
    measured_qubits = []
    measures = {}
    newlines = lines[:2]
    global_valid = True
    for line in lines[2:]:
        operations_qbs = line.split(" ")
        operations = operations_qbs[0]
        if operations == "qreg":
            newlines.append(line)
        elif operations == "creg":
            newlines.append(line)
        elif operations == "measure":
            mb = int(re.findall("\d+", operations_qbs[1])[0])
            cb = int(re.findall("\d+", operations_qbs[3])[0])
            measures[mb] = cb
            measured_qubits.append(mb)
            newlines.append(line)
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
                newlines.append(line)
    
    if not global_valid:
        print("Warning: All operations after measurement will be removed for executing on experiment") 
    new_openqasm = "\n".join(newlines)
    return new_openqasm

def openqasm_to_qLisp(openqasm):
    from numpy import pi
    import re
    lines = openqasm.splitlines()
    qlisp = []
    for line in lines[2:]:
        operations_qbs = line.split(" ")
        operations = operations_qbs[0]
        if operations == "qreg":
            qbs = operations_qbs[1]
            num = int(re.findall("\d+", qbs)[0])
        elif operations == "creg":
            pass
        elif operations == "measure":
            mb = int(re.findall("\d+", operations_qbs[1])[0])
            cb = int(re.findall("\d+", operations_qbs[3])[0])
            qlisp.append((("Measure", cb), "Q%d" %mb))
        else:
            qbs = operations_qbs[1]    
            indstr = re.findall("\d+", qbs)
            inds = [int(indst) for indst in indstr]  
            if operations == "barrier":
                qlisp.append(("Barrier", tuple(["Q%d" % i for i in inds])))
            else:
                sp_op = operations.split("(")
                gatename = sp_op[0]
                if len(sp_op) > 1:
                    paras = sp_op[1].strip("()")
                    parastr = paras.split(",")
                    paras = [eval(parai, {"pi":pi}) for parai in parastr]
                    
                if gatename == "cx":
                    qlisp.append(("Cnot", ("Q%d" %inds[0], "Q%d" %inds[1])))
                elif gatename == "cz":
                    qlisp.append(("CZ", ("Q%d" %inds[0], "Q%d" %inds[1])))
                elif gatename == "Rx":
                    qlisp.append((("Ry", *paras), "Q%d" % inds[0]))
                elif gatename == "ry":
                    qlisp.append((("Ry", *paras), "Q%d" % inds[0]))
                elif gatename == "rz":
                    qlisp.append((("Rz", *paras), "Q%d" % inds[0]))
                elif gatename in ("xyzh"):
                    qlisp.append((gatename.upper(), "Q%d" % inds[0]))
                elif gatename in ["u1", "u2", "u3"]:
                    qlisp.append(((gatename, *paras), "Q%d" % inds[0]))

    return str(qlisp)

print("input\n", test, "\n")
newopenqasm = cut_openqasm(test)
print("\n output\n", newopenqasm)
print("\nto qlisp\n", openqasm_to_qLisp(newopenqasm))
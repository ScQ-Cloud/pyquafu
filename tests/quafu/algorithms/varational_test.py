from quafu.elements import Parameter
from quafu.simulators.simulator import SVSimulator
from quafu import QuantumCircuit
from quafu.algorithms.hamiltonian import Hamiltonian, PauliOp, PauliMats
from quafu.elements.element_gates import RXGate, RYGate, RZGate, HGate, CXGate, CRYGate
from quafu.algorithms.gradient import grad_adjoint, grad_finit_diff, grad_para_shift
import scipy.sparse as sp
import numpy as np
import time
import math

class TestVariational:
    def test_autograd(self):
        a = Parameter("v1", 1.)
        b = Parameter("v2", 2.)
        c =  -a*b + 2. * b.log()
        assert math.isclose(c.value, -0.6137056388801094)
        assert math.isclose(c.get_value(), -0.6137056388801094)
        assert math.isclose(c.grad()[0], -2.) 
        assert math.isclose(c.grad()[1], 0.)

    def test_gates(self):
        q = QuantumCircuit(4)
        theta1 = Parameter("theta_1", 1.)
        theta2 = Parameter("theta_2", 2.)
        q << RXGate(0, theta1)
        q << RYGate(1, theta2)
        q.draw_circuit()

        backend = SVSimulator()
        res = backend.run(q)
        print(res["statevector"])

    def test_parameter_circuit(self):
        pq = QuantumCircuit(4)
        theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]
        for i in range(4):
            pq << RXGate(i, theta[i])
        
        pq << RYGate(2, theta[0]*theta[1]-3.*theta[0])
        pq.draw_circuit()
        print(pq.get_parameter_grads())

        variables = pq.variables
        print(variables)
        variables[0].value += np.pi/2
        assert variables[0] is theta[0]
        assert pq.gates[0].paras[0] is theta[0]
        pq.draw_circuit()

    def test_paulis(self):
        op1 = PauliOp("X2 Z0")
        mat1 = op1.get_matrix(4, True).toarray()
        mat2 = sp.kron(sp.kron(PauliMats["Z"], PauliMats["I"]), PauliMats["X"])
        mat2 = sp.kron(mat2, PauliMats["I"])
        mat2 = mat2.toarray()
        assert math.isclose(np.linalg.norm(mat1-mat2), 0.)

        mat1 = op1.get_matrix(4, False).toarray()
        mat2 = sp.kron(sp.kron(PauliMats["I"], PauliMats["X"]), PauliMats["I"])
        mat2 = sp.kron(mat2, PauliMats["Z"])
        mat2 = mat2.toarray()
        assert math.isclose(np.linalg.norm(mat1-mat2), 0.)


    def test_gradient(self):
        pq = QuantumCircuit(4)
        theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]
        for i in range(4):
            pq << HGate(i)
        for i in range(3):
            pq << CXGate(i, i+1)
        for i in range(4):
            pq << RYGate(i, theta[i])
        for i in range(3):
            pq << CXGate(i, i+1)
        for i in range(4):
            pq << RYGate(i, theta[i])

        pq << RYGate(2, theta[0]*theta[1]-3*theta[0])
        pq.draw_circuit()
        print(pq.get_parameter_grads())
        hamil = Hamiltonian([PauliOp("X1 X2"), PauliOp("Z0 Z3")])
        grads_fd = grad_finit_diff(pq, hamil)
        grads_ps = grad_para_shift(pq, hamil)
        grads_ad = grad_adjoint(pq, hamil)

        print(grads_fd)
        print(grads_ps)
        print(grads_ad)

        for i in range(len(grads_fd)):
            assert math.isclose(grads_ad[i], grads_ps[i])
            assert (abs(grads_fd[i]-grads_ps[i]) <  1e-5)

        pq._update_params([0.1, 0.73, 2.1, 3.2])
        grads_fd = grad_finit_diff(pq, hamil)
        grads_ps = grad_para_shift(pq, hamil)
        grads_ad = grad_adjoint(pq, hamil)
        for i in range(len(grads_fd)):
            assert math.isclose(grads_ad[i], grads_ps[i])
            assert (abs(grads_fd[i]-grads_ps[i]) <  1e-5)

        print(grads_fd)
        print(grads_ps)
        print(grads_ad)

    def test_ctrl_adjoint(self):
        pq = QuantumCircuit(4)
        theta = [Parameter("theta_%d" %(i), i+1) for i in range(4)]
        for i in range(4):
            pq << HGate(i)
        for i in range(3):
            pq << CXGate(i, i+1)
            pq << CRYGate(i, i+1,  theta[i])
        for i in range(3):
            pq << CXGate(i, i+1)
        for i in range(3):
            pq << CRYGate(i+1, i,  theta[i])

        hamil = Hamiltonian([PauliOp("X1 X2"), PauliOp("Z0 Z3")])
        grads_fd = grad_finit_diff(pq, hamil)
        grads_ad = grad_adjoint(pq, hamil)
        for i in range(len(grads_fd)):
           assert (abs(grads_fd[i]-grads_ad[i]) <  1e-5)

    def test_vqe(self):
        from scipy.optimize import minimize
        n = 5
        d = 6
        g = 0.5
        pq = QuantumCircuit(n)
        x0 = np.random.rand(n*d*3)
        theta = np.array([Parameter("theta_%d" %(i),  x0[i]) for i in range(n*d*3)])
        theta = np.reshape(theta, [d, n, 3])

        for j in range(d):
            for i in range(n):
                pq << RZGate(i, theta[j, i, 0])
                pq << RYGate(i, theta[j, i, 1])
                pq << RZGate(i, theta[j, i, 2])

            for i in range(n):
                pq << CXGate(i, (i+1)%n)

        

        # pq << RY(2, theta[0]*theta[1]-3*theta[0])
        pq.get_parameter_grads()
        ising_terms = [PauliOp(f"Z{j} Z{j+1}", -1.) for j in range(n-1)]
        ising_terms.extend([PauliOp(f"X{j}", g) for j in range(n)])
        hamil = Hamiltonian(ising_terms)
        h_mat =  hamil.get_matrix(n)
        from numpy.linalg import eigvalsh
        h_mat = h_mat.toarray()
        exact = eigvalsh(h_mat)[0]
        print("exact: ", exact)
        backend = SVSimulator()
        def cost(x, qc, hamil, backend):
            qc._update_params(x)
            return sum(backend.run(qc, hamiltonian=hamil)["pauli_expects"])

        def grad(x, qc, hamil, backend):
            qc._update_params(x)
            return grad_adjoint(qc, hamil)

        def wrap_cost(qc, hamil, backend):
            def callback(x):
                 global history
                 history.append(cost(x, qc, hamil, backend))
        
            return callback

        # sol = minimize(cost, x0, (pq, hamil, backend), "BFGS", grad, callback=wrap_cost(pq, hamil, backend))
        # print(sol)    
        # print(cost(sol.x, pq, hamil, backend))
        # import matplotlib.pyplot as plt

        # plt.plot(history)
        # plt.axhline(exact, linestyle='--', color='k')

        from quafu.algorithms.optimizer import adam
        sol, f, traj = adam(cost, x0, grad, (pq, hamil, backend), verbose=True, maxiter=300)
        print(f)
        assert abs(f-exact) < 0.01
        import matplotlib.pyplot as plt
        plt.plot(traj)
        plt.axhline(exact, linestyle='--', color='k')
        plt.xlabel("iteration")
        plt.ylabel("E")

    def test_vqe_with_wrap(self):
        from scipy.optimize import minimize
        n = 5
        d = 6
        g = 0.5
        pq = QuantumCircuit(n)
        x0 = np.random.rand(n*d*3)
        theta = np.array([Parameter("theta_%d" %(i),  x0[i]) for i in range(n*d*3)])
        theta = np.reshape(theta, [d, n, 3])

        linear_entangler = QuantumCircuit(n, name="linear-layer")
        for i in range(n):
            linear_entangler << CXGate(i, (i+1)%n)
        linear_entangler = linear_entangler.wrap()
        
        def u3_layer(thetas):
            _u3_layer = QuantumCircuit(n, name="linear-layer")
            for i in range(n):
                _u3_layer << RZGate(i, thetas[i, 0]) << RYGate(i, thetas[i, 1]) \
                << RZGate(i, thetas[i, 2])

            return _u3_layer.wrap()
        
        for j in range(d):
            pq << u3_layer(theta[j])
            pq << linear_entangler

        # pq << RY(2, theta[0]*theta[1]-3*theta[0])
        pq.draw_circuit()
        pq.get_parameter_grads()
        pq.draw_circuit()
        ising_terms = [PauliOp(f"Z{j} Z{j+1}", -1.) for j in range(n-1)]
        ising_terms.extend([PauliOp(f"X{j}", g) for j in range(n)])
        hamil = Hamiltonian(ising_terms)
        h_mat =  hamil.get_matrix(n)
        from numpy.linalg import eigvalsh
        h_mat = h_mat.toarray()
        exact = eigvalsh(h_mat)[0]
        print("exact: ", exact)
        backend = SVSimulator()
        def cost(x, qc, hamil, backend):
            qc._update_params(x)
            return sum(backend.run(qc, hamiltonian=hamil)["pauli_expects"])
        
        def grad(x, qc, hamil, backend):
            qc._update_params(x)
            return grad_adjoint(qc, hamil)

        def wrap_cost(qc, hamil, backend):
            def callback(x):
                global history
                history.append(cost(x, qc, hamil, backend))
            
            return callback
        
        # sol = minimize(cost, x0, (pq, hamil, backend), "BFGS", grad, callback=wrap_cost(pq, hamil, backend))
        # print(sol)    
        # print(cost(sol.x, pq, hamil, backend))
        # import matplotlib.pyplot as plt

        # plt.plot(history)
        # plt.axhline(exact, linestyle='--', color='k')

        from quafu.algorithms.optimizer import adam
        sol, f, traj = adam(cost, x0, grad, (pq, hamil, backend), verbose=True, maxiter=300)
        
        assert abs(f-exact) < 0.01
        import matplotlib.pyplot as plt
        plt.plot(traj)
        plt.axhline(exact, linestyle='--', color='k')
        plt.xlabel("iteration")
        plt.ylabel("E")

if __name__ == "__main__":
    TestVariational().test_vqe_with_wrap()
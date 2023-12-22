import unittest

import quafu.elements as qe
import quafu.elements.element_gates as qeg
import quafu.elements.element_gates.rotation
from numpy import pi


class TestGate(unittest.TestCase):
    def test_instances(self):
        gate_classes = qe.QuantumGate.gate_classes

        # Paulis
        x = qeg.XGate(pos=0)
        y = qeg.YGate(pos=0)
        z = qeg.ZGate(pos=0)
        i = qeg.IdGate(pos=0)
        w = qeg.WGate(pos=0)
        sw = qeg.SWGate(pos=0)
        swdg = qeg.SWdgGate(pos=0)
        sx = qeg.SXGate(pos=0)
        sxdg = qeg.SXdgGate(pos=0)
        sy = qeg.SYGate(pos=0)
        sydg = qeg.SYdgGate(pos=0)

        # Clifford
        h = qeg.HGate(pos=0)
        s = qeg.SGate(pos=0)
        sdg = qeg.SdgGate(pos=0)
        t = qeg.TGate(pos=0)
        tdg = qeg.TdgGate(pos=0)

        # Rotation
        ph = quafu.elements.element_gates.rotation.PhaseGate(pos=0, paras=pi)
        rx = qeg.RXGate(pos=0, paras=pi)
        ry = qeg.RYGate(pos=0, paras=pi)
        rz = qeg.RZGate(pos=0, paras=pi)
        rxx = qeg.RXXGate(q1=0, q2=3, paras=pi)
        ryy = qeg.RYYGate(q1=0, q2=3, paras=pi)
        rzz = qeg.RZZGate(q1=0, q2=3, paras=pi)

        # Swap
        swap = qeg.SwapGate(q1=0, q2=3)
        iswap = qeg.ISwapGate(q1=0, q2=3)
        fredkin = qeg.FredkinGate(ctrl=0, targ1=1, targ2=2)

        # Control
        cx = qeg.CXGate(ctrl=0, targ=1)
        cy = qeg.CYGate(ctrl=0, targ=1)
        cz = qeg.CZGate(ctrl=0, targ=1)
        cs = qeg.CSGate(ctrl=0, targ=1)
        ct = qeg.CTGate(ctrl=0, targ=1)
        cp = qeg.CPGate(ctrl=0, targ=1, paras=pi)
        mcx = qeg.MCXGate(ctrls=[0, 1, 2], targ=3)
        mcy = qeg.MCYGate(ctrls=[0, 1, 2], targ=3)
        mcz = qeg.MCZGate(ctrls=[0, 1, 2], targ=3)
        toffoli = qeg.ToffoliGate(ctrl1=0, ctrl2=1, targ=2)

        all_gates = [
            x,
            y,
            z,
            i,
            w,
            sw,
            swdg,
            sx,
            sxdg,
            sy,
            sydg,
            h,
            s,
            sdg,
            t,
            tdg,
            ph,
            rx,
            ry,
            rz,
            rxx,
            ryy,
            rzz,
            swap,
            iswap,
            fredkin,
            cx,
            cy,
            cz,
            cs,
            ct,
            cp,
            mcx,
            mcy,
            mcz,
            toffoli,
        ]
        self.assertEqual(len(all_gates), len(gate_classes))
        for gate in all_gates:
            self.assertIn(gate.name.lower(), gate_classes)


# TODO: test plots
# for gate in all_gates:
#     print(gate.name)
#     qc = QuantumCircuit(4)
#     qc.add_gate(gate)
#     qc.measure()
#     qc.plot_circuit(title=gate.__class__.__name__)
#     plt.savefig('./icons/%s.png' % gate.name, dpi=400, transparent=True)
#     plt.close()

if __name__ == "__main__":
    unittest.main()

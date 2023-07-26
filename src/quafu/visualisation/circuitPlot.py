import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection, PatchCollection, LineCollection
from matplotlib.patches import Circle, Arc
from matplotlib.text import Text

from quafu.elements.quantum_element import Instruction, ControlledGate

# the following line for developers only
# from quafu.circuits.quantum_circuit import QuantumCircuit

line_args = {}
box_args = {}

DEEPCOLOR = '#0C161F'
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
GOLDEN = '#FFB240'
GARNET = '#C0392B'

"""
layers(zorder):

0: figure
1: bkg box
2: wires
3: closed patches
4: white bkg for label/text
5: labels
"""

su2_gate_names = ['x', 'y', 'z', 'id', 'w',
                  'h', 't', 'tdg', 's', 'sdg', 'sx', 'sy', 'sw',
                  'phase',
                  'rx', 'ry', 'rz',
                  ]

swap_gate_names = ['swap', 'iswap']
r2_gate_names = ['rxx', 'ryy', 'rzz']
c2_gate_names = ['cp', 'cs', 'ct', 'cx', 'cy', 'cz']
c3_gate_names = ['fredkin', 'toffoli']
mc_gate_names = ['mcx', 'mcy', 'mcz']
operation_names = ['barrier', 'delay']


class CircuitPlotManager:
    """
    A class to manage the plotting of quantum circuits.
    Stores style parameters and provides functions to plot.

    To be initialized when circuit.plot() is called.
    """
    # colors
    _wire_color = '#FF0000'
    _light_blue = '#3B82F6'
    _ec = DEEPCOLOR

    _wire_lw = 1.5

    _a_inch = 2 / 2.54  # physical lattice constant in inch
    _a = 0.5  # box width and height, unit: ax
    _barrier_width = _a / 3  # barrier width

    _stroke = pe.withStroke(linewidth=2, foreground='white')

    def __init__(self, qc):
        """
        Processing graphical info from a quantum circuit, decompose every
        gate/instruction into graphical elements and send the latter into
        corresponding containers.

        At present quantum gates and barriers are stored as a list. In the
        near future the circuit will be stored as a graph or graph-like object,
        procedure will be much simplified, and more functions will be developed.
        (TODO)
        """
        # step0: containers of graphical elements

        self._h_wire_points = []
        self._ctrl_wire_points = []

        self._closed_patches = []
        self._mea_arc_patches = []
        self._mea_point_patches = []

        self._ctrl_points = []
        self._not_points = []
        self._swap_points = []
        self._iswap_points = []
        self._barrier_points = []

        self._text_list = []

        # step1: process gates/instructions
        self.dorders = np.zeros(qc.num, dtype=int)
        for gate in qc.gates:
            assert isinstance(gate, Instruction)
            self._process_ins(gate)
        qubits_used = self.dorders > 0
        self.used_qbit_num = qc.num
        # TODO: drop off unused-qubits
        # self.used_qbit_num = np.sum(qubits_used)

        self.depth = np.max(self.dorders) + 1

        for q, c in qc.measures.items():
            self._proc_measure(self.depth - 1, q)

        # step2: initialize bit-label
        self.q_label = {i: r'$|q_{%d}\rangle$' % i for i in range(qc.num) if qubits_used[i]}
        self.c_label = {iq: f'c_{ic}' for iq, ic in qc.measures.items() if qubits_used[iq]}

        # step3: figure coordination
        self.xs = np.arange(-3 / 2, self.depth + 3 / 2)
        self.ys = np.arange(-2, self.used_qbit_num + 1 / 2)

    def __call__(self,
                 title=None,
                 init_labels=None,
                 end_labels=None,
                 save_path: str = None,
                 show: bool = False,
                 *args,
                 **kwargs):
        """
        :param title
        :param init_labels: dict, {qbit: label}
        :param end_labels: dict, {qbit: label}
        :param save_path: str, path to save the figure
        :param show: bool, whether to show the figure
        :param args:
        :param kwargs:

        More customization will be supported in the future.(TODO)
        """
        # Not supported by patch collections?
        # if 'xkcd' in kwargs:
        #     import random
        #     plt.gca().xkcd(randomness=random.randint(0, 1000))
        if title is not None:
            title = Text((self.xs[0] + self.xs[-1]) / 2, -0.8,
                         title,
                         size=30,
                         ha='center', va='baseline')
            self._text_list.append(title)

        # initialize a figure
        _size_x = self._a_inch * abs(self.xs[-1] - self.xs[0])
        _size_y = self._a_inch * abs(self.ys[-1] - self.ys[0])
        fig = plt.figure(figsize=(_size_x, _size_y))  # inch
        ax = fig.add_axes([0, 0, 1, 1],
                          aspect=1,
                          xlim=[self.xs[0], self.xs[-1]],
                          ylim=[self.ys[0], self.ys[-1]],
                          )
        ax.axis('off')
        ax.invert_yaxis()

        self._circuit_wires()
        self._inits_label(labels=init_labels)
        self._measured_label(labels=end_labels)
        self._render_circuit()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def _process_ins(self, ins: Instruction, append: bool = True):
        name = ins.name
        assert name in Instruction.ins_classes, 'If this should occur, please report a bug.'

        name = name.lower()
        _which = slice(np.min(ins.pos), np.max(ins.pos) + 1)
        depth = np.max(self.dorders[_which])
        paras = ins.paras

        if name == 'barrier':
            self._proc_barrier(depth, ins.pos)
        elif name == 'measure':
            self._proc_measure(depth, ins.pos)
        elif name in su2_gate_names:
            self._proc_su2(name, depth, ins.pos, paras)
        elif name in swap_gate_names:
            self._proc_swap(depth, ins.pos, name == 'iswap')
        elif name in r2_gate_names:
            # TODO: combine into one box
            self._proc_su2(name[-1], depth, ins.pos[0], paras)
            self._proc_su2(name[-1], depth, ins.pos[1], paras)
        elif isinstance(ins, ControlledGate):
            self._proc_ctrl(depth, ins)
        else:
            raise NotImplementedError(f'Gate {name} is not supported yet.\n'
                                      f'If this should occur, please report a bug.')
        if append:
            self.dorders[_which] = depth + 1

    #########################################################################
    # Helper functions for processing gates/instructions into graphical
    # elements. Add only points data of for the following collection-wise
    # plotting if possible, create a patch otherwise.
    #########################################################################
    def _circuit_wires(self):
        """
        plot horizontal circuit wires
        """
        for y in range(self.used_qbit_num):
            x0 = self.xs[0] + 1
            x1 = self.xs[-1] - 1
            self._h_wire_points.append([[x0, y], [x1, y]])

    def _gate_bbox(self, x, y, fc: str):
        """ Single qubit gate box """
        a = self._a
        from matplotlib.patches import FancyBboxPatch
        bbox = FancyBboxPatch((-a / 2 + x, -a / 2 + y), a, a,  # this warning belongs to matplotlib
                              boxstyle=f'round, pad={0.2 * a}',
                              edgecolor=DEEPCOLOR,
                              facecolor=fc,
                              )
        self._closed_patches.append(bbox)

    def _inits_label(self, labels: dict[int: str] = None):
        """ qubit-labeling """
        if labels is None:
            labels = self.q_label

        for i, label in labels.items():
            txt = Text(-2 / 3, i,
                       label,
                       size=18,
                       color=DEEPCOLOR,
                       ha='right',
                       va='center',
                       )
            self._text_list.append(txt)

    def _measured_label(self, labels: dict[int: str] = None):
        """ measured qubit-labeling """
        if labels is None:
            labels = self.c_label

        for i, label in labels.items():
            label = r'$%s$' % label
            txt = Text(self.xs[-1] - 3 / 4, i,
                       label,
                       size=18,
                       color=DEEPCOLOR,
                       ha='left',
                       va='center',
                       )
            self._text_list.append(txt)

    def _gate_label(self, s, x, y):
        if not s:
            return None
        _dy = 0.05
        text = Text(x, y + _dy,
                    s,
                    size=24,
                    color=DEEPCOLOR,
                    ha='center',
                    va='center',
                    )
        text.set_path_effects([self._stroke])
        self._text_list.append(text)

    def _para_label(self, para_txt, x, y):
        """ label parameters """
        if not para_txt:
            return None
        _dx = 0
        text = Text(x + _dx, y + 0.8 * self._a,
                    para_txt,
                    size=12,
                    color=DEEPCOLOR,
                    ha='center',
                    va='top',
                    )
        self._text_list.append(text)

    def _measure_label(self, x, y):
        from matplotlib.patches import FancyArrow
        a = self._a
        r = 1.1 * a
        d = 1.2 * a / 3.5

        arrow = FancyArrow(x=x,
                           y=y + d,
                           dx=0.15,
                           dy=-0.35,
                           width=0.04,
                           facecolor=DEEPCOLOR,
                           head_width=0.07,
                           head_length=0.15,
                           edgecolor='white')
        arc = Arc((x, y + d),
                  width=r,
                  height=r,
                  lw=1,
                  theta1=180,
                  theta2=0,
                  fill=False,
                  zorder=4,
                  color=DEEPCOLOR,
                  capstyle='round',
                  )
        center_bkg = Circle((x, y + d),
                            radius=0.035,
                            color='white',
                            )
        center = Circle((x, y + d),
                        radius=0.025,
                        facecolor=DEEPCOLOR,
                        )
        self._mea_arc_patches.append(arc)
        self._mea_point_patches += [center_bkg, arrow, center]

    #########################################################################
    # # # # processing-functions: decompose ins into graphical elements # # #
    #########################################################################
    def _proc_su2(self, id_name, depth, pos, paras):
        if id_name in ['x', 'y', 'z', 'h', 'id', 's', 't', 'p', 'u']:
            fc = '#EE7057'
            label = id_name.capitalize()
        elif id_name in ['rx', 'ry', 'rz']:
            fc = '#6366F1'
            label = id_name.upper()
        else:
            fc = '#8C9197'
            label = '?'

        if id_name in ['rx', 'ry', 'rz', 'p']:
            # too long to display: r'$\theta=$' + f'{paras:.3f}' (TODO)
            para_txt = f'({paras:.3f})'
        else:
            para_txt = None

        self._gate_label(label, depth, pos)
        self._para_label(para_txt, depth, pos)
        self._gate_bbox(depth, pos, fc)

    def _proc_ctrl(self, depth, ins: ControlledGate, ctrl_type: bool = True):
        # control part
        p0, p1 = np.max(ins.pos), np.min(ins.pos)
        self._ctrl_wire_points.append([[depth, p1], [depth, p0]])

        ctrl_pos = np.array(ins.ctrls)
        for c in ctrl_pos:
            self._ctrl_points.append((depth, c, ctrl_type))

        # target part
        name = ins.name.lower()
        if ins.ct_nums == (1, 1, 2) or name in mc_gate_names:
            tar_name = ins.targ_name.lower()[-1]
            pos = ins.targs if isinstance(ins.targs, int) else ins.targs[0]
            if tar_name == 'x':
                self._not_points.append((depth, pos))
            else:
                self._proc_su2(tar_name, depth, pos, None)
        elif name == 'cswap':
            self._swap_points += [[depth, p] for p in ins.targs]
        elif name == 'ccx':
            self._not_points.append((depth, ins.targs))
        else:
            from quafu.elements.element_gates import ControlledU
            assert isinstance(ins, ControlledU), f'unknown gate: {name}, {ins.__class__.__name__}'
            self._process_ins(ins, append=False)

    def _proc_swap(self, depth, pos, iswap: bool = False):
        p1, p2 = pos
        nodes = [[depth, p] for p in pos]
        self._swap_points += nodes
        self._ctrl_wire_points.append([[depth, p1], [depth, p2]])
        if iswap:
            self._iswap_points += nodes

    def _proc_barrier(self, depth, pos: list):
        x0 = depth - self._barrier_width
        x1 = depth + self._barrier_width

        for p in pos:
            y0 = (p - 1 / 2)
            y1 = (p + 1 / 2)
            nodes = [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]
            self._barrier_points.append(nodes)

    def _proc_measure(self, depth, pos):
        fc = GOLDEN
        self._gate_bbox(depth, pos, fc)
        self._measure_label(depth, pos)

        # TODO: decide whether to draw double wire for measurement
        # y = pos + 0.02
        # x0 = depth
        # x1 = self.depth - 1 / 2
        # self._h_wire_points.append([[x0, y], [x1, y]])

    #########################################################################
    # # # # # # # # # # # # # # rendering functions # # # # # # # # # # # # #
    #########################################################################
    def _render_h_wires(self):
        h_lines = LineCollection(self._h_wire_points,
                                 zorder=0,
                                 colors=self._wire_color,
                                 alpha=0.8,
                                 linewidths=2,
                                 )
        plt.gca().add_collection(h_lines)

    def _render_ctrl_wires(self):
        v_lines = LineCollection(self._ctrl_wire_points,
                                 zorder=0,
                                 colors=self._light_blue,
                                 alpha=0.8,
                                 linewidths=4,
                                 )
        plt.gca().add_collection(v_lines)

    def _render_closed_patch(self):
        collection = PatchCollection(self._closed_patches,
                                     match_original=True,
                                     zorder=3,
                                     ec=self._ec,
                                     linewidths=0.5,
                                     )
        plt.gca().add_collection(collection)

    def _render_ctrl_nodes(self):
        circle_collection = []
        r = self._a / 4
        for x, y, ctrl in self._ctrl_points:
            fc = '#3B82F6' if ctrl else 'white'
            circle = Circle((x, y), radius=r, fc=fc)
            circle_collection.append(circle)
        circles = PatchCollection(circle_collection,
                                  match_original=True,
                                  zorder=5,
                                  ec=self._ec,
                                  linewidths=2,
                                  )
        plt.gca().add_collection(circles)

    def _render_not_nodes(self):
        points = []
        rp = self._a * 0.3
        r = self._a * 0.5

        for x, y in self._not_points:
            points.append([[x, y - rp], [x, y + rp]])
            points.append([[x - rp, y], [x + rp, y]])
            circle = Circle((x, y), radius=r, lw=1,
                            fc='#3B82F6')
            self._closed_patches.append(circle)

        collection = LineCollection(points,
                                    zorder=5,
                                    colors='white',
                                    linewidths=2,
                                    capstyle='round',
                                    )
        plt.gca().add_collection(collection)

    def _render_swap_nodes(self):
        points = []
        r = self._a / (4 ** (1 / 2))
        for x, y in self._swap_points:
            points.append([[x - r, y - r], [x + r, y + r]])
            points.append([[x + r, y - r], [x - r, y + r]])
        collection = LineCollection(points,
                                    zorder=5,
                                    colors='#3B82F6',
                                    linewidths=4,
                                    capstyle='round',
                                    )
        plt.gca().add_collection(collection)

        # iswap-cirlces
        i_circles = []
        for x, y in self._iswap_points:
            circle = Circle((x, y), radius=2 ** (1 / 2) * r, lw=3,
                            ec='#3B82F6', fill=False)
            i_circles.append(circle)
        collection = PatchCollection(i_circles,
                                     match_original=True,
                                     zorder=5,
                                     )
        plt.gca().add_collection(collection)

    def _render_measure(self):
        stroke = pe.withStroke(linewidth=4, foreground='white')
        arcs = PatchCollection(self._mea_arc_patches,
                               match_original=True,
                               capstyle='round',
                               zorder=4)
        arcs.set_path_effects([stroke])

        plt.gca().add_collection(arcs)
        pointers = PatchCollection(self._mea_point_patches,  # note the order
                                   match_original=True,
                                   zorder=5,
                                   facecolors=DEEPCOLOR,
                                   linewidths=2,
                                   )
        plt.gca().add_collection(pointers)

    def _render_barrier(self):
        barrier = PolyCollection(self._barrier_points,
                                 closed=True,
                                 fc='lightgray',
                                 hatch='///',
                                 zorder=4)
        plt.gca().add_collection(barrier)

    def _render_txt(self):
        for txt in self._text_list:
            plt.gca().add_artist(txt)

    def _render_circuit(self):
        self._render_h_wires()
        self._render_ctrl_wires()
        self._render_ctrl_nodes()
        self._render_not_nodes()

        self._render_swap_nodes()
        self._render_measure()
        self._render_barrier()
        self._render_closed_patch()
        self._render_txt()

quafu.core.gates
======================

.. automodule:: quafu.core.gates

Base Class
-------------

.. autosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.BasicGate
    quafu.core.gates.NoneParameterGate
    quafu.core.gates.ParameterGate
    quafu.core.gates.QuantumGate
    quafu.core.gates.NoiseGate

Quantum Gate
-------------

.. msmathautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.CNOTGate
    quafu.core.gates.FSim
    quafu.core.gates.GlobalPhase
    quafu.core.gates.HGate
    quafu.core.gates.IGate
    quafu.core.gates.ISWAPGate
    quafu.core.gates.Measure
    quafu.core.gates.PhaseShift
    quafu.core.gates.Rn
    quafu.core.gates.RX
    quafu.core.gates.Rxx
    quafu.core.gates.Rxy
    quafu.core.gates.Rxz
    quafu.core.gates.RY
    quafu.core.gates.Ryy
    quafu.core.gates.Ryz
    quafu.core.gates.RZ
    quafu.core.gates.Rzz
    quafu.core.gates.RotPauliString
    quafu.core.gates.SGate
    quafu.core.gates.SWAPGate
    quafu.core.gates.SWAPalpha
    quafu.core.gates.SXGate
    quafu.core.gates.TGate
    quafu.core.gates.U3
    quafu.core.gates.XGate
    quafu.core.gates.YGate
    quafu.core.gates.ZGate
    quafu.core.gates.GroupedPauli
    quafu.core.gates.Givens

Functional Gate
----------------

.. autosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.UnivMathGate
    quafu.core.gates.gene_univ_parameterized_gate
    quafu.core.gates.BarrierGate

pre-instantiated gate
----------------------

The gates blow are the pre-instantiated quantum gates, which can be used directly as an instance of quantum gate.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - pre-instantiated gate
     - gate
   * - quafu.core.gates.CNOT
     - :class:`quafu.core.gates.CNOTGate`
   * - quafu.core.gates.I
     - :class:`quafu.core.gates.IGate`
   * - quafu.core.gates.ISWAP
     - :class:`quafu.core.gates.ISWAPGate`
   * - quafu.core.gates.H
     - :class:`quafu.core.gates.HGate`
   * - quafu.core.gates.S
     - :class:`quafu.core.gates.PhaseShift` (numpy.pi/2)
   * - quafu.core.gates.SWAP
     - :class:`quafu.core.gates.SWAPGate`
   * - quafu.core.gates.SX
     - :class:`quafu.core.gates.SXGate`
   * - quafu.core.gates.T
     - :class:`quafu.core.gates.PhaseShift` (numpy.pi/4)
   * - quafu.core.gates.X
     - :class:`quafu.core.gates.XGate`
   * - quafu.core.gates.Y
     - :class:`quafu.core.gates.YGate`
   * - quafu.core.gates.Z
     - :class:`quafu.core.gates.ZGate`

Quantum Channel
----------------

.. msmathautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.AmplitudeDampingChannel
    quafu.core.gates.BitFlipChannel
    quafu.core.gates.BitPhaseFlipChannel
    quafu.core.gates.DepolarizingChannel
    quafu.core.gates.KrausChannel
    quafu.core.gates.PauliChannel
    quafu.core.gates.GroupedPauliChannel
    quafu.core.gates.PhaseDampingChannel
    quafu.core.gates.PhaseFlipChannel
    quafu.core.gates.ThermalRelaxationChannel

Functional Class
-----------------

.. autosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.MeasureResult
    quafu.core.gates.Power

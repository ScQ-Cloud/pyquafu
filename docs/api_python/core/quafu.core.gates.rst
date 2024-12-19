quafu.core.gates
======================

.. py:module:: quafu.core.gates


量子门模块，提供不同的量子门。

基类
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.BasicGate
    quafu.core.gates.NoneParameterGate
    quafu.core.gates.ParameterGate
    quafu.core.gates.QuantumGate
    quafu.core.gates.NoiseGate

通用量子门
-------------

.. mscnmathautosummary::
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
    quafu.core.gates.SWAPalpha
    quafu.core.gates.SWAPGate
    quafu.core.gates.SXGate
    quafu.core.gates.TGate
    quafu.core.gates.U3
    quafu.core.gates.XGate
    quafu.core.gates.YGate
    quafu.core.gates.ZGate
    quafu.core.gates.GroupedPauli
    quafu.core.gates.Givens

功能量子门
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.UnivMathGate
    quafu.core.gates.gene_univ_parameterized_gate
    quafu.core.gates.gene_univ_two_params_gate
    quafu.core.gates.BarrierGate

预实例化门
----------

如下量子门是预实例化的量子门，可直接作为对应量子门的实例来使用。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - gates
   * - quafu.core.gates.CNOT
     - :class:`~.core.gates.CNOTGate`
   * - quafu.core.gates.I
     - :class:`~.core.gates.IGate`
   * - quafu.core.gates.ISWAP
     - :class:`~.core.gates.ISWAPGate`
   * - quafu.core.gates.H
     - :class:`~.core.gates.HGate`
   * - quafu.core.gates.S
     - :class:`~.core.gates.PhaseShift` (numpy.pi/2)
   * - quafu.core.gates.SWAP
     - :class:`~.core.gates.SWAPGate`
   * - quafu.core.gates.SX
     - :class:`~.core.gates.SXGate`
   * - quafu.core.gates.T
     - :class:`~.core.gates.PhaseShift` (numpy.pi/4)
   * - quafu.core.gates.X
     - :class:`~.core.gates.XGate`
   * - quafu.core.gates.Y
     - :class:`~.core.gates.YGate`
   * - quafu.core.gates.Z
     - :class:`~.core.gates.ZGate`

量子信道
-------------

.. mscnmathautosummary::
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

功能类
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.gates.MeasureResult
    quafu.core.gates.Power

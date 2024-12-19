quafu.core.circuit
========================

.. py:module:: quafu.core.circuit


量子线路模块，通过有序地组织各种量子门，我们可以轻松地搭建出符合要求的量子线路，包括参数化量子线路。本模块还包含各种预设的量子线路以及对量子线路进行高效操作的模块。

Class
---------------

.. mscnautosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.circuit.Circuit
    quafu.core.circuit.SwapParts
    quafu.core.circuit.UN

Function
---------------

.. mscnautosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.circuit.add_prefix
    quafu.core.circuit.add_suffix
    quafu.core.circuit.apply
    quafu.core.circuit.as_ansatz
    quafu.core.circuit.as_encoder
    quafu.core.circuit.change_param_name
    quafu.core.circuit.controlled
    quafu.core.circuit.dagger
    quafu.core.circuit.decompose_single_term_time_evolution
    quafu.core.circuit.pauli_word_to_circuits
    quafu.core.circuit.shift
    quafu.core.circuit.qfi
    quafu.core.circuit.partial_psi_partial_psi
    quafu.core.circuit.partial_psi_psi

Channel adder
-------------

.. mscnautosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.circuit.ChannelAdderBase
    quafu.core.circuit.NoiseChannelAdder
    quafu.core.circuit.MeasureAccepter
    quafu.core.circuit.ReverseAdder
    quafu.core.circuit.NoiseExcluder
    quafu.core.circuit.BitFlipAdder
    quafu.core.circuit.MixerAdder
    quafu.core.circuit.SequentialAdder
    quafu.core.circuit.QubitNumberConstrain
    quafu.core.circuit.QubitIDConstrain
    quafu.core.circuit.GateSelector
    quafu.core.circuit.DepolarizingChannelAdder

functional
----------

如下的操作符是对应量子线路操作符的简写。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - high level circuit operators
   * - quafu.core.circuit.C
     - :class:`~.core.circuit.controlled`
   * - quafu.core.circuit.D
     - :class:`~.core.circuit.dagger`
   * - quafu.core.circuit.A
     - :class:`~.core.circuit.apply`
   * - quafu.core.circuit.AP
     - :class:`~.core.circuit.add_prefix`
   * - quafu.core.circuit.CPN
     - :class:`~.core.circuit.change_param_name`

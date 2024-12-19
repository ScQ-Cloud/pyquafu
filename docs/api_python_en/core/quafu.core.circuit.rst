quafu.core.circuit
========================

.. automodule:: quafu.core.circuit

Class
---------------

.. autosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.circuit.Circuit
    quafu.core.circuit.SwapParts
    quafu.core.circuit.UN

Function
---------------

.. autosummary::
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

.. autosummary::
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

shortcut
----------

The operators blow are shortcut of correspand quantum circuit operators.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - shortcut
     - high level circuit operators
   * - quafu.core.circuit.C
     - :class:`quafu.core.circuit.controlled`
   * - quafu.core.circuit.D
     - :class:`quafu.core.circuit.dagger`
   * - quafu.core.circuit.A
     - :class:`quafu.core.circuit.apply`
   * - quafu.core.circuit.AP
     - :class:`quafu.core.circuit.add_prefix`
   * - quafu.core.circuit.CPN
     - :class:`quafu.core.circuit.change_param_name`

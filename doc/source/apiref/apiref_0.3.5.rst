API Reference
=============

.. note::

    This page is still under refinement, the contents you see may be incomplete at present.

Quantum Circuit
------------------

.. autoclass:: quafu.QuantumCircuit
    :members: cnot, add_gate, barrier
    :undoc-members: add_pulse

Quantum Elements
------------------
.. hint::
    hello

.. autoclass:: quafu.elements.quantum_element.Instruction
    :members:


Task and User
------------------

.. autoclass:: quafu.Task
    :members: config, send, retrieve

.. autoclass:: quafu.User
    :members:

.. autoclass:: quafu.results.results.Result
    :members:

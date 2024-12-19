quafu.core.operators
==========================

.. py:module:: quafu.core.operators


quafu算子库。算子由一个或多个基本门的组合而成。

包含以下类的表示：

- Qubit算子

- Fermion算子

- 时间演化算子

Class
---------------

.. mscnautosummary::
    :toctree: operators
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.operators.FermionOperator
    quafu.core.operators.Hamiltonian
    quafu.core.operators.InteractionOperator
    quafu.core.operators.PolynomialTensor
    quafu.core.operators.Projector
    quafu.core.operators.QubitExcitationOperator
    quafu.core.operators.QubitOperator
    quafu.core.operators.TimeEvolution

Function
---------------

.. mscnautosummary::
    :toctree: operators
    :nosignatures:
    :template: classtemplate.rst

    quafu.core.operators.commutator
    quafu.core.operators.count_qubits
    quafu.core.operators.down_index
    quafu.core.operators.get_fermion_operator
    quafu.core.operators.ground_state_of_sum_zz
    quafu.core.operators.hermitian_conjugated
    quafu.core.operators.normal_ordered
    quafu.core.operators.number_operator
    quafu.core.operators.sz_operator
    quafu.core.operators.up_index

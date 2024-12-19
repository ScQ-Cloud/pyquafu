quafu.algorithm.compiler
==============================

.. py:module:: quafu.algorithm.compiler


quafu 量子线路编译模块。

Fixed decompose rules
---------------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.compiler.ch_decompose
    quafu.algorithm.compiler.crx_decompose
    quafu.algorithm.compiler.crxx_decompose
    quafu.algorithm.compiler.cry_decompose
    quafu.algorithm.compiler.cnry_decompose
    quafu.algorithm.compiler.crz_decompose
    quafu.algorithm.compiler.cnrz_decompose
    quafu.algorithm.compiler.cryy_decompose
    quafu.algorithm.compiler.cswap_decompose
    quafu.algorithm.compiler.ct_decompose
    quafu.algorithm.compiler.cy_decompose
    quafu.algorithm.compiler.cz_decompose
    quafu.algorithm.compiler.rxx_decompose
    quafu.algorithm.compiler.ryy_decompose
    quafu.algorithm.compiler.rzz_decompose
    quafu.algorithm.compiler.cs_decompose
    quafu.algorithm.compiler.swap_decompose
    quafu.algorithm.compiler.ccx_decompose

Universal decompose rules
-------------------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.compiler.euler_decompose
    quafu.algorithm.compiler.cu_decompose
    quafu.algorithm.compiler.qs_decompose
    quafu.algorithm.compiler.abc_decompose
    quafu.algorithm.compiler.kak_decompose
    quafu.algorithm.compiler.tensor_product_decompose

Compiler rules
--------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.compiler.BasicCompilerRule
    quafu.algorithm.compiler.KroneckerSeqCompiler
    quafu.algorithm.compiler.SequentialCompiler
    quafu.algorithm.compiler.BasicDecompose
    quafu.algorithm.compiler.CZBasedChipCompiler
    quafu.algorithm.compiler.CXToCZ
    quafu.algorithm.compiler.CZToCX
    quafu.algorithm.compiler.GateReplacer
    quafu.algorithm.compiler.FullyNeighborCanceler
    quafu.algorithm.compiler.SimpleNeighborCanceler
    quafu.algorithm.compiler.compile_circuit

DAG circuit
-----------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.compiler.DAGCircuit
    quafu.algorithm.compiler.DAGNode
    quafu.algorithm.compiler.GateNode
    quafu.algorithm.compiler.DAGQubitNode
    quafu.algorithm.compiler.connect_two_node
    quafu.algorithm.compiler.try_merge

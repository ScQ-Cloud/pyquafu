quafu.dtype
=================

.. py:module:: quafu.dtype


quafu 数据类型模拟。

支持的数据类型
-------------------

如下类型是 quafu 在进行量子模拟时支持的类型。

.. list-table::
   :widths: 50 50

   * - quafu.float32
     - single precision real number type
   * - quafu.float64
     - double precision real number type
   * - quafu.complex64
     - single precision complex number type
   * - quafu.complex128
     - double precision complex number type

Memory consuming
-------------------

下表展示全振幅量子态内存占用与比特数的关系:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - qubit number
     - complex128
     - complex64
   * - 6
     - 1kB
     - 0.5kB
   * - 16
     - 1MB
     - 0.5MB
   * - 26
     - 1GB
     - 0.5GB
   * - 30
     - 16GB
     - 8GB
   * - 36
     - 1TB
     - 0.5TB
   * - 40
     - 16TB
     - 8TB
   * - 46
     - 1PB
     - 0.5PB

Function
---------------

.. mscnautosummary::
    :toctree: dtype
    :nosignatures:
    :template: classtemplate.rst

    quafu.dtype.is_double_precision
    quafu.dtype.is_single_precision
    quafu.dtype.is_same_precision
    quafu.dtype.precision_str
    quafu.dtype.to_real_type
    quafu.dtype.to_complex_type
    quafu.dtype.to_single_precision
    quafu.dtype.to_double_precision
    quafu.dtype.to_precision_like
    quafu.dtype.to_quafu_type
    quafu.dtype.to_np_type

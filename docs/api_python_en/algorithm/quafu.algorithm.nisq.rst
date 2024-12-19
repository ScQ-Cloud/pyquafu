quafu.algorithm.nisq
===========================

.. automodule:: quafu.algorithm.nisq

Base Class
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.nisq.Ansatz

Encoder
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.nisq.IQPEncoding
    quafu.algorithm.nisq.QuantumNeuron

Ansatz
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.nisq.HardwareEfficientAnsatz
    quafu.algorithm.nisq.Max2SATAnsatz
    quafu.algorithm.nisq.MaxCutAnsatz
    quafu.algorithm.nisq.QubitUCCAnsatz
    quafu.algorithm.nisq.StronglyEntangling
    quafu.algorithm.nisq.UCCAnsatz

.. toctree::
    :hidden:

    nisq/quafu.algorithm.nisq.RYLinear
    nisq/quafu.algorithm.nisq.RYFull
    nisq/quafu.algorithm.nisq.RYCascade
    nisq/quafu.algorithm.nisq.RYRZFull
    nisq/quafu.algorithm.nisq.PCHeaXYZ1F
    nisq/quafu.algorithm.nisq.PCHeaXYZ2F
    nisq/quafu.algorithm.nisq.ASWAP

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Images
    * - :class:`quafu.algorithm.nisq.RYLinear`
      - .. image:: nisq/ansatz_images/RYLinear.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.RYFull`
      - .. image:: nisq/ansatz_images/RYFull.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.RYCascade`
      - .. image:: nisq/ansatz_images/RYCascade.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.RYRZFull`
      - .. image:: nisq/ansatz_images/RYRZFull.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.PCHeaXYZ1F`
      - .. image:: nisq/ansatz_images/PCHeaXYZ1F.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.PCHeaXYZ2F`
      - .. image:: nisq/ansatz_images/PCHeaXYZ2F.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.ASWAP`
      - .. image:: nisq/ansatz_images/ASWAP.png
            :height: 180px

The following Ansatz come from paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_.

.. toctree::
    :hidden:

    nisq/quafu.algorithm.nisq.Ansatz1
    nisq/quafu.algorithm.nisq.Ansatz2
    nisq/quafu.algorithm.nisq.Ansatz3
    nisq/quafu.algorithm.nisq.Ansatz4
    nisq/quafu.algorithm.nisq.Ansatz5
    nisq/quafu.algorithm.nisq.Ansatz6
    nisq/quafu.algorithm.nisq.Ansatz7
    nisq/quafu.algorithm.nisq.Ansatz8
    nisq/quafu.algorithm.nisq.Ansatz9
    nisq/quafu.algorithm.nisq.Ansatz10
    nisq/quafu.algorithm.nisq.Ansatz11
    nisq/quafu.algorithm.nisq.Ansatz12
    nisq/quafu.algorithm.nisq.Ansatz13
    nisq/quafu.algorithm.nisq.Ansatz14
    nisq/quafu.algorithm.nisq.Ansatz15
    nisq/quafu.algorithm.nisq.Ansatz16
    nisq/quafu.algorithm.nisq.Ansatz17
    nisq/quafu.algorithm.nisq.Ansatz18
    nisq/quafu.algorithm.nisq.Ansatz19

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Images
    * - :class:`quafu.algorithm.nisq.Ansatz1`
      - .. image:: nisq/ansatz_images/ansatz1.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz2`
      - .. image:: nisq/ansatz_images/ansatz2.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz3`
      - .. image:: nisq/ansatz_images/ansatz3.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz4`
      - .. image:: nisq/ansatz_images/ansatz4.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz5`
      - .. image:: nisq/ansatz_images/ansatz5.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz6`
      - .. image:: nisq/ansatz_images/ansatz6.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz7`
      - .. image:: nisq/ansatz_images/ansatz7.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz8`
      - .. image:: nisq/ansatz_images/ansatz8.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz9`
      - .. image:: nisq/ansatz_images/ansatz9.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz10`
      - .. image:: nisq/ansatz_images/ansatz10.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz11`
      - .. image:: nisq/ansatz_images/ansatz11.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz12`
      - .. image:: nisq/ansatz_images/ansatz12.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz13`
      - .. image:: nisq/ansatz_images/ansatz13.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz14`
      - .. image:: nisq/ansatz_images/ansatz14.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz15`
      - .. image:: nisq/ansatz_images/ansatz15.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz16`
      - .. image:: nisq/ansatz_images/ansatz16.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz17`
      - .. image:: nisq/ansatz_images/ansatz17.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz18`
      - .. image:: nisq/ansatz_images/ansatz18.png
            :height: 180px
    * - :class:`quafu.algorithm.nisq.Ansatz19`
      - .. image:: nisq/ansatz_images/ansatz19.png
            :height: 180px

Generator
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.nisq.generate_uccsd
    quafu.algorithm.nisq.quccsd_generator
    quafu.algorithm.nisq.uccsd0_singlet_generator
    quafu.algorithm.nisq.uccsd_singlet_generator

Functional
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    quafu.algorithm.nisq.Transform
    quafu.algorithm.nisq.get_qubit_hamiltonian
    quafu.algorithm.nisq.uccsd_singlet_get_packed_amplitudes
    quafu.algorithm.nisq.ansatz_variance
    quafu.algorithm.nisq.get_reference_circuit

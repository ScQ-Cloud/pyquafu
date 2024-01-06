Release Notes - v0.3.6

- Fixed bug of status code mixed up during receiving results.
- Added attributes ``named_paras`` and ``named_pos`` into ``Instruction`` and all of its subclasses. This is to further unify interface of instructions. Added ``instructions`` into ``QuantumCircuit``, the former existing ``gates`` will contain only instances of ``QuantumGate`` in future.
- Fixed some icon-missing bugs in ``CircuitPlotManager``, plots of all basic gates was tested.

发布说明 - v0.3.6

- 修复了接收结果时状态码混淆的bug。
- 添加了属性 ``named_paras`` 和``named_pos`` 到 ``Instruction`` 和它的所有子类。这是为了进一步统一 ``Instruction`` 的接口。在 ``QuantumCircuit``中添加属性 ``instructions`` ，先前存在的``gates``在稍后的版本中将只包含``QuantumGate`` 的实例。
- 修复了``CircuitPlotManager``中一些图标缺失的bug，测试了所有基本门的绘图。
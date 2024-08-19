from typing import List, Union
from quafu.transpile.passes.basepass import BasePass


class PassFlow:
    """A quafu.transpile passflow describes a quantum circuit compilation process."""

    def __init__(self, passes: Union[List[BasePass], BasePass]) -> None:
        if isinstance(passes, BasePass):
            passes = [passes]

        self._passes = list(passes)

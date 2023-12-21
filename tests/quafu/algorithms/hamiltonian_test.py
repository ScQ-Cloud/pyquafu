# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from quafu.algorithms.hamiltonian import Hamiltonian

M_0 = np.array(
    [
        [
            3.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            3.0 + 0.0j,
        ],
    ]
)

M_1 = np.array(
    [
        [
            0.0 + 0.0j,
            2.0 + 0.0j,
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            2.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -2.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            -2.0 + 0.0j,
            0.0 + 0.0j,
        ],
    ]
)

M_2 = np.array(
    [
        [
            0.0 + 0.0j,
            0.0 - 1.0j,
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 1.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            1.0 + 0.0j,
            0.0 + 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 1.0j,
        ],
        [
            0.0 + 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 1.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
            0.0 + 0.0j,
        ],
    ]
)


class TestHamiltonian:
    def test_init(self):
        h = Hamiltonian(["IIIZZ", "IIZIZ"], np.array([1, 1]))
        h = Hamiltonian.from_pauli_list(
            [("IIIZZ", 1), ("IIZIZ", 1), ("IZIIZ", 1), ("ZIIIZ", 1)]
        )

    def test_to_matrix(self):
        h = Hamiltonian.from_pauli_list([("IZZ", 1), ("ZIZ", 1), ("ZZI", 1)])
        m = h.get_matrix()
        assert np.array_equal(m, M_0)

        h = Hamiltonian.from_pauli_list([("IZX", 1), ("ZIX", 1), ("ZXI", 1)])
        m = h.get_matrix()
        assert np.array_equal(m, M_1)

        h = Hamiltonian.from_pauli_list([("YZI", 1), ("ZIY", 1), ("ZXI", 1)])
        m = h.get_matrix()
        assert np.array_equal(m, M_2)

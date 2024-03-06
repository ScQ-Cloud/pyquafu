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

"""Amplitude Embedding by a decomposition into gates"""
import quafu.elements.element_gates as qeg
import numpy as np
from quafu.elements import QuantumGate


# from .basic_entangle import BasicEntangleLayers


class AmplitudeEmbedding:
    def __init__(self, state, num_qubits, pad_with=None, normalize=False):
        """
        Args:
            state(np.array): The state to be embedded
            num_qubits(int): the number of qubit
            pad_with (float or complex): if not None, the input will be padded to size 2**num_qubits
            normalize (bool): whether to automatically normalize the state
        """
        self.num_qubits = num_qubits
        self.pad_with = pad_with
        self.normalize = normalize
        self.state = self._preprocess(state, num_qubits, pad_with, normalize)
        self.gate_list = self._build()

    def __iter__(self):
        return iter(self.gate_list)

    def __getitem__(self, index):
        return self.gate_list[index]
    
    def __add__(self, gates):
        """Addition operator."""
        out = []
        out.extend(self.gate_list)
        if all(isinstance(gate, QuantumGate) for gate in gates):
            out.extend(gates)
        else:
            raise TypeError("Contains unsupported gate")
        return out
    
    def __radd__(self, other):
        out = []
        if all(isinstance(gate, QuantumGate) for gate in other):
            out.extend(other)
        else:
            raise TypeError("Contains unsupported gate")
        out.extend(self.gate_list)
        return out

    def _preprocess(self, state, num_qubits, pad_with, normalize):
        batched = np.ndim(state) > 1
        ##TODO(qtzhuang): If state are batched, additional processing is required
        if batched:
            raise ValueError("Currently not support batched state.")
        state_batch = state if batched else [state]
        new_state_batch = []

        # apply pre-processing
        for feature_set in state_batch:
            shape = np.shape(feature_set)

            # check shape
            if len(shape) != 1:
                raise ValueError(f"state must be a one-dimensional tensor; got shape {shape}.")

            n_state = shape[0]
            dim = 2 ** num_qubits
            if pad_with is None and n_state != dim:
                raise ValueError(
                    f"The length of state should be {dim}; got length {n_state}.Please check num_qubits "
                    f"or Use the 'pad_with' argument for automated padding."
                )

            if pad_with is not None:
                if n_state > dim:
                    raise ValueError(
                        f"state must be of length {dim} or smaller to be padded; got length {n_state}."
                    )

                # pad
                if n_state < dim:
                    padding = [pad_with] * (dim - n_state)
                    padding = np.asarray(padding, dtype=feature_set.dtype)
                    feature_set = np.hstack([feature_set, padding])

            # normalize
            norm = np.sum(np.abs(feature_set) ** 2)
            # tolerance for normalization
            TOLERANCE = 1e-10
            if not np.allclose(norm, 1.0, atol=TOLERANCE):
                if normalize or pad_with:
                    feature_set = feature_set / np.sqrt(norm)
                else:
                    raise ValueError(
                        f"state must be a vector of norm 1.0; got norm {norm}. "
                        "Use 'normalize=True' to automatically normalize."
                    )
            new_state_batch.append(feature_set)

        return np.stack(new_state_batch).astype(np.complex128) if batched else new_state_batch[0].astype(np.complex128)

    def _build(self):  

        a = np.abs(self.state)
        omega = np.angle(self.state)
        # change order of qubits, since original code was written for IBM machines
        qubits_reverse = range(self.num_qubits)[::-1]
        gate_list = []

        # Apply inverse y rotation cascade to prepare correct absolute values of amplitudes
        for k in range(len(qubits_reverse), 0, -1):
            alpha_y_k = _get_alpha_y(a, len(qubits_reverse), k)
            control = qubits_reverse[k:]
            target = qubits_reverse[k - 1]
            gate_list.extend(_apply_uniform_rotation_dagger(qeg.RYGate, alpha_y_k, control, target))

        # If necessary, apply inverse z rotation cascade to prepare correct phases of amplitudes
        if not np.allclose(omega, 0):
            for k in range(len(qubits_reverse), 0, -1):
                alpha_z_k = _get_alpha_z(omega, len(qubits_reverse), k)
                control = qubits_reverse[k:]
                target = qubits_reverse[k - 1]
                if len(alpha_z_k) > 0:
                    gate_list.extend(
                        _apply_uniform_rotation_dagger(qeg.RZGate, alpha_z_k, control, target)
                    )

        return gate_list
    
## MottonenStatePreparation related functions.
def gray_code(rank):
    """Generates the Gray code of given rank.

    Args:
        rank (int): rank of the Gray code (i.e. number of bits)
    """

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return

        for i in range(k - 1, -1, -1):
            char = "1" + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = "0" + g[i]

        gray_code_recurse(g, rank - 1)

    g = ["0", "1"]
    gray_code_recurse(g, rank - 1)

    return g

def _matrix_M_entry(row, col):
        
        # (col >> 1) ^ col is the Gray code of col
        b_and_g = row & ((col >> 1) ^ col)
        sum_of_ones = 0
        while b_and_g > 0:
            if b_and_g & 0b1:
                sum_of_ones += 1

            b_and_g = b_and_g >> 1

        return (-1) ** sum_of_ones

def compute_theta(alpha):
    ln = alpha.shape[-1]
    k = np.log2(ln)

    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
        for j in range(len(M_trans[0])):
            M_trans[i, j] = _matrix_M_entry(j, i)

    theta = np.transpose(np.dot(M_trans, np.transpose(alpha)))

    return theta / 2**k


def _apply_uniform_rotation_dagger(gate, alpha, control_wires, target_wire):

    gate_list = []
    theta = compute_theta(alpha)

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        if np.all(theta[..., 0] != 0.0):
            gate_list.append(gate(target_wire, theta[0]))
        return gate_list

    code = gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    for i, control_index in enumerate(control_indices):
        if np.all(theta[..., i] != 0.0):
            gate_list.append(gate(target_wire, theta[i]))
        gate_list.append(qeg.CXGate(control_wires[control_index], target_wire))
    return gate_list

def _get_alpha_z(omega, n, k):

    indices1 = [
        [(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]
    indices2 = [
        [(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]

    term1 = np.take(omega, indices=indices1, axis=-1)
    term2 = np.take(omega, indices=indices2, axis=-1)
    diff = (term1 - term2) / 2 ** (k - 1)

    return np.sum(diff, axis=-1)

def _get_alpha_y(a, n, k):

    indices_numerator = [
        [(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))]
        for j in range(2 ** (n - k))
    ]
    numerator = np.take(a, indices=indices_numerator, axis=-1)
    numerator = np.sum(np.abs(numerator) ** 2, axis=-1)

    indices_denominator = [[j * 2**k + l for l in range(2**k)] for j in range(2 ** (n - k))]
    denominator = np.take(a, indices=indices_denominator, axis=-1)
    denominator = np.sum(np.abs(denominator) ** 2, axis=-1)

    # Divide only where denominator is zero, else leave initial value of zero.
    # The equation guarantees that the numerator is also zero in the corresponding entries.

    with np.errstate(divide="ignore", invalid="ignore"):
        division = numerator / denominator

    # Cast the numerator and denominator to ensure compatibility with interfaces
    division = np.array(division, dtype=np.float64)
    denominator = np.array(denominator, dtype=np.float64)

    division = np.where(denominator != 0.0, division, 0.0)

    return 2 * np.arcsin(np.sqrt(division))
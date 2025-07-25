# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
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
"""Quantum pulse module."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from .instruction import Instruction

TimeType = Union[np.ndarray, float, int]


# pylint: disable=too-many-arguments,too-many-positional-arguments
class QuantumPulse(Instruction, ABC):
    pulse_classes = {}

    def __init__(
        self,
        pos: int,
        paras: list,
        duration: Union[float, int],
        unit: str,
        channel: str,
    ):
        """
        Quantum Pulse for generating a quantum gate.

        Args:
            pos (int): Qubit position.
            paras (list): Parameters of the pulse.
            duration (float, int): Pulse duration.
            unit (str): Duration unit.
        """
        super().__init__()  # pylint: disable=no-value-for-parameter
        self.pos = pos
        self.paras = paras
        self.duration = duration
        self.unit = unit
        if channel in ["XY", "Z"]:
            self.channel = channel
        else:
            raise ValueError("channel must be 'XY' or 'Z'")

    @property
    def symbol(self):
        return f"{self.name}({self.duration}{self.unit}, {self.channel})"

    @abstractmethod
    def time_func(self, t: Union[np.ndarray, float, int], **kwargs):
        """
        Return the pulse data.

        Args:
            t (np.ndarray, float, int): Time list.
            kwargs (dict): Keyword arguments for the pulse.
        """

    @classmethod
    def register_pulse(cls, subclass, name: str = None):
        assert issubclass(subclass, cls)

        if name is None:
            name = subclass.name
        if name in cls.pulse_classes:
            raise ValueError(f"Name {name} already exists.")
        cls.pulse_classes[name] = subclass
        Instruction.register_ins(subclass, name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        symbol = f"{self.name}({self.duration}{self.unit}"
        for para in self.paras:
            symbol += f", {para}"
        symbol += f", {self.channel}"
        symbol += ")"
        return symbol

    def __call__(
        self,
        t: TimeType,
        shift: Union[float, int] = 0.0,
        offset: Union[float, int] = 0.0,
        args: dict = None,
    ):
        """
        Return pulse data.

        Args:
            t (np.ndarray, float, int): Time list.
            shift (float, int): Time shift.
            offset (float, int): Pulse amplitude offset.
        """
        window = np.logical_and(0 <= t, t <= self.duration)
        if args is None:
            return window * self.time_func(t - shift)
        return window * self.time_func(t - shift, **args)

    def __copy__(self):
        """Return a deepcopy of the pulse"""
        return deepcopy(self)

    def to_qasm(self, _):
        return str(self) + f" q[{self.pos}]"

    def plot(
        self,
        t: Optional[np.ndarray] = None,
        shift: Union[float, int] = 0.0,
        offset: Union[float, int] = 0.0,
        plot_real: bool = True,
        plot_imag: bool = True,
        fig=None,
        ax=None,
        **plot_kws,
    ):
        """
        Plot the pulse waveform.

        Args:
            t (np.ndarray): Time list of the plot.
            shift (float, int): Time shift of the pulse.
            offset (float, int): Offset of the pulse.
            plot_real (bool): Plot real of the pulse.
            plot_imag (bool): Plot imag of the pulse.
            fig (Figure): Figure of the plot.
            ax (Axes): Axes of the plot.
            plot_kws (dict, optional): Plot kwargs of `ax.plot`.
        """
        if t is None:
            t = np.linspace(0, self.duration, 101)
        if ax is None:
            fig, ax = plt.subplots(1, 1, num=fig)
        pulse_data = self(t, shift=shift, offset=offset)
        if plot_real:
            ax.plot(np.real(pulse_data), label="real", **plot_kws)
        if plot_imag:
            ax.plot(np.imag(pulse_data), label="imag", **plot_kws)
        ax.set_xlabel(f"Time ({self.unit})")
        ax.set_ylabel("Pulse Amp (a.u.)")
        ax.legend()
        plt.show()

    def set_pos(self, pos: int):
        """Set qubit position"""
        self.pos = pos
        return self

    def set_unit(self, unit="ns"):
        """Set duration unit"""
        self.unit = unit
        return self


class RectPulse(QuantumPulse):
    name = "rect"

    def __init__(self, pos, amp, duration, unit, channel):
        self.amp = amp

        super().__init__(pos, [amp], duration, unit, channel)

    def time_func(self, t: Union[np.ndarray, float, int], **kwargs):
        """rect_time_func"""
        amp_ = kwargs["amp"]
        return amp_ * np.ones(np.array(t).shape)

    def __call__(
        self,
        t: TimeType,
        *args,
        shift: Union[float, int] = 0,
        offset: Union[float, int] = 0,
        **kwargs,
    ):
        args = {"amp": self.amp}
        return super().__call__(t, shift, offset, args)


class FlattopPulse(QuantumPulse):
    name = "flattop"

    def __init__(self, pos, amp, fwhm, duration, unit, channel):
        self.amp = amp
        self.fwhm = fwhm

        super().__init__(pos, [amp, fwhm], duration, unit, channel)

    def time_func(self, t, **kws):
        """flattop_time_func"""
        amp_, fwhm_ = kws["amp"], kws["fwhm"]
        sigma_ = fwhm_ / (2 * np.sqrt(np.log(2)))
        return amp_ * (erf((self.duration - t) / sigma_) + erf(t / sigma_) - 1.0)

    def __call__(
        self,
        t: TimeType,
        *args,
        shift: Union[float, int] = 0,
        offset: Union[float, int] = 0,
        **kwargs,
    ):
        args = {"amp": self.amp, "fwhm": self.fwhm}
        return super().__call__(t, shift, offset, args)


class GaussianPulse(QuantumPulse):
    name = "gaussian"

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, pos, amp, fwhm, phase, duration, unit, channel):
        self.amp = amp
        if fwhm is None:
            self.fwhm = 0.5 * duration
        else:
            self.fwhm = fwhm
        self.phase = phase

        super().__init__(pos, [amp, fwhm, phase], duration, unit, channel)

    def time_func(self, t, **kws):
        """gaussian_time_func"""
        amp_, fwhm_, phase_ = kws["amp"], kws["fwhm"], kws["phase"]
        # start: t = 0, center: t = 0.5 * duration, end: t = duration
        sigma_ = fwhm_ / np.sqrt(8 * np.log(2))  # fwhm to std. deviation
        return amp_ * np.exp(
            -((t - 0.5 * self.duration) ** 2) / (2 * sigma_**2) + 1j * phase_
        )

    def __call__(
        self,
        t: TimeType,
        *args,
        shift: Union[float, int] = 0,
        offset: Union[float, int] = 0,
        **kwargs,
    ):
        args = {"amp": self.amp, "fwhm": self.fwhm, "phase": self.phase}
        return super().__call__(t, shift, offset, args)


class Delay(Instruction):
    name = "delay"

    def __init__(self, pos: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        super().__init__(pos)
        self.unit = unit
        self.symbol = f"Delay({duration}{unit})"

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self, _):
        return f"delay({self.duration}{self.unit}) q[{self.pos}]"

    @property
    def named_paras(self) -> Dict:
        return {"duration": self.duration}

    @property
    def named_pos(self) -> Dict:
        return {"pos": self.pos}


class XYResonance(Instruction):
    name = "XY"

    def __init__(self, qs: int, qe: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        super().__init__(list(range(qs, qe + 1)))
        self.unit = unit
        self.symbol = f"XY({duration}{unit})"

    def to_qasm(self, _):
        return f"xy({self.duration}{self.unit}) " + ",".join(
            [f"q[{p}]" for p in range(min(self.pos), max(self.pos) + 1)]
        )


QuantumPulse.register_pulse(RectPulse)
QuantumPulse.register_pulse(FlattopPulse)
QuantumPulse.register_pulse(GaussianPulse)
Instruction.register_ins(Delay)
Instruction.register_ins(XYResonance)

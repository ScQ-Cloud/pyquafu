from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from .instruction import Instruction, PosType

TimeType = Union[np.ndarray, float, int]


class QuantumPulse(Instruction, ABC):
    pulse_classes = {}

    def __init__(self,
                 pos: PosType,
                 duration: Union[float, int],
                 unit: str = 'ns',
                 channel: str = None,
                 paras: list = None,
                 ):
        """
        Quantum Pulse for generating a quantum gate.

        Args:
            pos (int): Qubit position.
            paras (list): Parameters of the pulse.
            duration (float, int): Pulse duration.
            unit (str): Duration unit.
        """
        super().__init__(pos, paras)
        self.duration = duration
        self.unit = unit
        if channel in ["XY", "Z"]:
            self.channel = channel
        else:
            raise ValueError("channel must be 'XY' or 'Z'")

    @property
    def symbol(self):
        return "%s(%d%s, %s)" % (self.name, self.duration, self.unit, self.channel)

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}

    @abstractmethod
    def time_func(self, t: Union[np.ndarray, float, int]):
        """
        Return the pulse data.

        Args:
            t (np.ndarray, float, int): Time list.
        """
        pass

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
        symbol = "%s(%d%s" % (self.name, self.duration, self.unit)
        for para in self.paras:
            symbol += ", %s" % para
        symbol += ", %s" % self.channel
        symbol += ")"
        return symbol

    def __call__(self,
                 t: TimeType,
                 shift: Union[float, int] = 0.,
                 offset: Union[float, int] = 0.
                 ):
        """
        Return pulse data.

        Args:
            t (np.ndarray, float, int): Time list.
            shift (float, int): Time shift.
            offset (float, int): Pulse amplitude offset.
        """
        window = np.logical_and(0 <= t, t <= self.duration)
        return window * self.time_func(t - shift)

    def __copy__(self):
        """ Return a deepcopy of the pulse """
        return deepcopy(self)

    def to_qasm(self):
        return self.__str__() + " q[%d]" % self.pos

    # TODO: deprecate this
    def plot(self,
             t: Optional[np.ndarray] = None,
             shift: Union[float, int] = 0.,
             offset: Union[float, int] = 0.,
             plot_real: bool = True,
             plot_imag: bool = True,
             fig=None,
             ax=None,
             **plot_kws):
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
            ax.plot(np.real(pulse_data), label='real', **plot_kws)
        if plot_imag:
            ax.plot(np.imag(pulse_data), label='imag', **plot_kws)
        ax.set_xlabel("Time (%s)" % self.unit)
        ax.set_ylabel("Pulse Amp (a.u.)")
        ax.legend()
        plt.show()

    def set_unit(self, unit="ns"):
        """ Set duration unit """
        self.unit = unit
        return self


class RectPulse(QuantumPulse):
    name = "rect"

    def __init__(self, pos, amp, duration, unit, channel):
        self.amp = amp

        super().__init__(pos, duration, unit, channel, amp)

    @property
    def named_paras(self) -> Dict:
        named_paras = {'amp': self.amp,
                       'duration': self.duration}
        return named_paras

    def time_func(self, t: Union[np.ndarray, float, int]):
        """ rect_time_func """
        return self.amp * np.ones(np.array(t).shape)


class FlattopPulse(QuantumPulse):
    name = "flattop"

    def __init__(self, pos, amp, fwhm, duration, unit, channel):
        self.amp = amp
        self.fwhm = fwhm

        super().__init__(pos, duration, unit, channel, [amp, fwhm])

    @property
    def named_paras(self) -> Dict:
        named_paras = {'amp': self.amp,
                       'duration': self.duration,
                       "fwhm": self.fwhm}
        return named_paras
    
    def time_func(self, t):
        """ flattop_time_func """
        from scipy.special import erf
        sigma_ = self.fwhm / (2 * np.sqrt(np.log(2)))
        return self.amp * (erf((self.duration - t) / sigma_) + erf(t / sigma_) - 1.)


class GaussianPulse(QuantumPulse):
    name = "gaussian"

    def __init__(self, pos, duration, unit, channel, amp, fwhm, phase):
        self.amp = amp
        self.fwhm = 0.5 * duration if fwhm is None else fwhm
        self.phase = phase

        super().__init__(pos, duration, unit, channel, [amp, fwhm, phase])

    def time_func(self, t):
        """ gaussian_time_func """
        # start: t = 0, center: t = 0.5 * duration, end: t = duration
        sigma_ = self.fwhm / np.sqrt(8 * np.log(2))  # fwhm to std. deviation
        return self.amp * np.exp(
            -(t - 0.5 * self.duration) ** 2 / (2 * sigma_ ** 2) + 1j * self.phase)

    @property
    def named_paras(self) -> Dict:
        named_paras = {'amp': self.amp,
                       'duration': self.duration,
                       "fwhm": self.fwhm,
                       "phase": self.phase}
        return named_paras


class Delay(Instruction):
    name = "delay"

    def __init__(self, pos: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        super().__init__(pos)
        self.unit = unit
        self.symbol = "Delay(%d%s)" % (duration, unit)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def to_qasm(self):
        return "delay(%d%s) q[%d]" % (self.duration, self.unit, self.pos)

    @property
    def named_paras(self) -> Dict:
        return {}

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}


class XYResonance(Instruction):
    name = "XY"

    def __init__(self, qs: int, qe: int, duration: int, unit="ns"):
        if isinstance(duration, int):
            self.duration = duration
        else:
            raise TypeError("duration must be int")
        super().__init__(list(range(qs, qe + 1)))
        self.unit = unit
        self.symbol = "XY(%d%s)" % (duration, unit)

    def to_qasm(self):
        return "xy(%d%s) " % (self.duration, self.unit) + ",".join(
            ["q[%d]" % p for p in range(min(self.pos), max(self.pos) + 1)])

    @property
    def named_pos(self) -> Dict:
        return {'pos': self.pos}

    @property
    def named_paras(self) -> Dict:
        return {'duration': self.duration}


QuantumPulse.register_pulse(RectPulse)
QuantumPulse.register_pulse(FlattopPulse)
QuantumPulse.register_pulse(GaussianPulse)
Instruction.register_ins(Delay)
Instruction.register_ins(XYResonance)

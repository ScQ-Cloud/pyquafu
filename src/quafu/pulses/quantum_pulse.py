# !/usr/bin/env python
# -*- coding:utf-8 -*-

# @File: quantum_pulse.py
# @Version: 1.0
# @Author: Yun-Hao Shi
# @Email: yhshi@iphy.ac.cn
# @Reminders: |0> has been in the past, |1> is still in the future
# @License: Copyright (c) 2023, IOP, CAS

"""

"""
from typing import Union, Optional, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import scipy.special


class QuantumPulse(object):
    def __init__(self,
                 pos: int,
                 duration: Union[float, int],
                 unit="ns",
                 name: str = "Pulse",
                 time_func: Optional[Callable] = None,
                 args: Optional[Dict] = None
                 ):
        """
        Quantum Pulse for generating a quantum gate.

        Args:
            pos (int): Qubit position.
            duration (float, int): Pulse duration.
            unit (str): Duration unit.
            name (str): Pulse name.
            time_func (callable): Time function of the pulse.
                Where t=0 is the start, t=duration is the end of the pulse.
            args (dict, optional): Other parameters of the pulse.
        """
        self.pos = pos
        self.duration = duration
        self.unit = unit
        self.name = name
        self.time_func = time_func
        self.args = args
        self.paras = [1, None, 0.]
        for v in self.args:
            if v == "amp":
                self.paras[0] = self.args["amp"]
            elif v == "fwhm":
                self.paras[1] = self.args["fwhm"]
            elif v == "phase":
                self.paras[2] = self.args["phase"]
                
    @property
    def pos(self) -> int:
        return self.__pos

    @pos.setter
    def pos(self, _pos: int):
        self.__pos = _pos

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, _name: str):
        self.__name = str(_name)

    @property
    def unit(self) -> str:
        return self.__unit

    @unit.setter
    def unit(self, _unit: str):
        self.__unit = str(_unit)

    @property
    def symbol(self):
        return "%s(%d%s)" % (self.name, self.duration, self.unit)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        symbol = "%s(%d%s" % (self.name, self.duration, self.unit)
        for para in self.paras:
            symbol += ", %s" %para
        symbol += ")"
        return symbol

    def __call__(self,
                 t: Union[np.ndarray, float, int],
                 shift: Union[float, int] = 0.,
                 offset: Union[float, int] = 0.):
        """
        Return pulse data.

        Args:
            t (np.ndarray, float, int): Time list.
            shift (float, int): Time shift.
            offset (float, int): Pulse amplitude offset.
        """
        window = np.logical_and(0 <= t, t <= self.duration)
        if self.args is None:
            return window * self.time_func(t - shift)
        else:
            return window * self.time_func(t - shift, **self.args)

    def __copy__(self):
        """ Return a deepcopy of the pulse """
        return deepcopy(self)

    def to_qasm(self):
        return self.__str__() + " q[%d]" %self.pos
    
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

    def set_pos(self, pos: int):
        """ Set qubit position """
        self.pos = pos
        return self

    def set_unit(self, unit="ns"):
        """ Set duration unit """
        self.unit = unit
        return self


def rect(pos: int,
         duration: Union[int, float],
         amp: Union[float, int],
         unit: str = "ns"):
    """
    Rectangular pulse.

    Args:
        pos (int): Qubit position.
        duration (float, int): Pulse duration.
        amp (float, int): Amplitude of the pulse.
        unit (str): Duration unit.
    """

    def rect_time_func(t, **kws):
        amp_ = kws["amp"]
        return amp_ * np.ones(np.array(t).shape)

    return QuantumPulse(pos=pos,
                        duration=duration,
                        unit=unit,
                        name="Rect",
                        time_func=rect_time_func,
                        args={"amp": float(amp)})


def flattop(pos: int,
            duration: Union[int, float],
            amp: Union[float, int],
            fwhm: Union[float, int] = 2,
            unit: str = "ns"):
    """
    Rectangular pulse with smooth (error-function-type) rise and fall.

    Args:
        pos (int): Qubit position.
        duration (float, int): Pulse duration.
        amp (float, int): Amplitude of the pulse.
        fwhm (float, int): Full width at half maximum.
        unit (str): Duration unit.
    """

    def flattop_time_func(t, **kws):
        amp_, fwhm_ = kws["amp"], kws["fwhm"]
        sigma_ = fwhm_ / (2 * np.sqrt(np.log(2)))
        return amp_ * (scipy.special.erf((duration - t) / sigma_)
                      + scipy.special.erf(t / sigma_) - 1.)

    return QuantumPulse(pos=pos,
                        duration=duration,
                        unit=unit,
                        name="Flattop",
                        time_func=flattop_time_func,
                        args={"amp": float(amp), "fwhm": float(fwhm)})


def gaussian(pos: int,
             duration: Union[int, float],
             amp: Union[float, int],
             fwhm: Optional[Union[float, int]] = None,
             phase: Union[float, int] = 0.,
             unit: str = "ns"):
    """
    Gaussian pulse.

    Args:
        pos (int): Qubit position.
        duration (float, int): Pulse duration.
        amp (float, int): Amplitude of the pulse.
        fwhm (float, int): Full width at half maximum, default: duration/2.
        phase (int, float): pulse phase.
        unit (str): Duration unit.
    """
    amp = float(amp)
    if fwhm is None:
        fwhm = 0.5 * duration

    def gaussian_time_func(t, **kws):
        amp_, fwhm_, phase_ = kws["amp"], kws["fwhm"], kws["phase"]
        # start: t = 0, center: t = 0.5 * duration, end: t = duration
        sigma_ = fwhm_ / np.sqrt(8 * np.log(2))  # fwhm to std. deviation
        return amp_ * np.exp(
            -(t - 0.5 * duration) ** 2 / (2 * sigma_ ** 2) + 1j * phase_)

    return QuantumPulse(pos=pos,
                        duration=duration,
                        unit=unit,
                        name="Gaussian",
                        time_func=gaussian_time_func,
                        args={"amp": amp, "fwhm": fwhm, "phase": phase})


# def main():
#     # g = gaussian(0, 60, 1.2, phase=np.pi / 3)
#     g = flattop(pos=0, duration=60, amp=1., fwhm=10)
#     g.plot(shift=0)


# if __name__ == '__main__':
#     main()

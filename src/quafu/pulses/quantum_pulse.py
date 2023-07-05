from typing import Union, Optional, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import scipy.special


class QuantumPulse(object):
    def __init__(self,
                 name: str,
                 pos: Union[int, list],
                 paras: list,
                 duration: Union[float, int],
                 unit: str,
                 channel: str,
                 time_func: Optional[Callable] = None,
                 ):
        """
        Quantum Pulse for generating a quantum gate.

        Args:
            name (str): Pulse name
            pos (int): Qubit position.
            paras (list): Parameters of the pulse.
            duration (float, int): Pulse duration.
            unit (str): Duration unit.
            name (str): Pulse name.
            time_func (callable): Time function of the pulse.
                Where t=0 is the start, t=duration is the end of the pulse.
            
        """

        self.name = name
        self.pos = pos
        self.paras = paras
        self.duration = duration
        self.unit = unit
        self.time_func = time_func
        if channel in ["XY", "Z"]:
            self.channel = channel
        else:
            raise ValueError("channel must be 'XY' or 'Z'")

    @property
    def symbol(self):
        return "%s(%d%s, %s)" % (self.name, self.duration, self.unit, self.channel)

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
                 t: Union[np.ndarray, float, int],
                 shift: Union[float, int] = 0.,
                 offset: Union[float, int] = 0.,
                 args: dict = None
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
        else:
            return window * self.time_func(t - shift, **args)

    def __copy__(self):
        """ Return a deepcopy of the pulse """
        return deepcopy(self)

    def to_qasm(self):
        return self.__str__() + " q[%d]" % self.pos

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


class RectPulse(QuantumPulse):
    def __init__(self, pos, amp, duration, unit, channel):
        self.amp = amp

        def rect_time_func(t, **kws):
            amp_ = kws["amp"]
            return amp_ * np.ones(np.array(t).shape)

        super().__init__("rect", pos, [amp], duration, unit, channel, rect_time_func)

    def __call__(self, t: Union[np.ndarray, float, int], shift: Union[float, int] = 0, offset: Union[float, int] = 0):
        args = {"amp": self.amp}
        return super().__call__(t, shift, offset, args)


class FlattopPulse(QuantumPulse):
    def __init__(self, pos, amp, fwhm, duration, unit, channel):
        self.amp = amp
        self.fwhm = fwhm

        def flattop_time_func(t, **kws):
            amp_, fwhm_ = kws["amp"], kws["fwhm"]
            sigma_ = fwhm_ / (2 * np.sqrt(np.log(2)))
            return amp_ * (scipy.special.erf((duration - t) / sigma_)
                           + scipy.special.erf(t / sigma_) - 1.)

        super().__init__("flattop", pos, [amp, fwhm], duration, unit, channel, flattop_time_func)

    def __call__(self, t: Union[np.ndarray, float, int], shift: Union[float, int] = 0, offset: Union[float, int] = 0):
        args = {"amp": self.amp, "fwhm": self.fwhm}
        return super().__call__(t, shift, offset, args)


class GaussianPulse(QuantumPulse):
    def __init__(self, pos, amp, fwhm, phase, duration, unit, channel):
        self.amp = amp
        if fwhm == None:
            self.fwhm = 0.5 * duration
        else:
            self.fwhm = fwhm

        self.phase = phase

        def gaussian_time_func(t, **kws):
            amp_, fwhm_, phase_ = kws["amp"], kws["fwhm"], kws["phase"]
            # start: t = 0, center: t = 0.5 * duration, end: t = duration
            sigma_ = fwhm_ / np.sqrt(8 * np.log(2))  # fwhm to std. deviation
            return amp_ * np.exp(
                -(t - 0.5 * duration) ** 2 / (2 * sigma_ ** 2) + 1j * phase_)

        super().__init__("gaussian", pos, [amp, fwhm, phase], duration, unit, channel, gaussian_time_func)

    def __call__(self, t: Union[np.ndarray, float, int], shift: Union[float, int] = 0, offset: Union[float, int] = 0):
        args = {"amp": self.amp, "fwhm": self.fwhm, "phase": self.phase}
        return super().__call__(t, shift, offset, args)

from .pauli import XGate, YGate, ZGate, HGate, IdGate, WGate, SWGate
from .phase import PhaseGate
from .rotation import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate
from .swap import SwapGate, iSwapGate
from .sqrt import SXGate, SXdgGate, SYGate, SYdgGate, SGate, SdgGate, TGate, TdgGate
from .c11 import CXGate, CYGate, CZGate, CSGate, CTGate, CPGate
from .c21 import ToffoliGate
from .c12 import FredkinGate
from .cm1 import MCXGate, MCYGate, MCZGate

__all__ = ['XGate', 'YGate', 'ZGate', 'HGate', 'IdGate', 'WGate', 'SWGate',
           'SXGate', 'SXdgGate', 'SYGate', 'SYdgGate', 'SGate', 'SdgGate', 'TGate', 'TdgGate',
           'PhaseGate',
           'RXGate', 'RYGate', 'RZGate', 'RXXGate', 'RYYGate', 'RZZGate',
           'SwapGate', 'iSwapGate',
           'CXGate', 'CYGate', 'CZGate', 'CSGate', 'CTGate', 'CPGate',
           'ToffoliGate',
           'FredkinGate',
           'MCXGate', 'MCYGate', 'MCZGate']

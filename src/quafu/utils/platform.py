import os
import sys
from ..exceptions import QuafuError


def get_homedir():
    if sys.platform == 'win32':
        homedir = os.environ['USERPROFILE']
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        homedir = os.environ['HOME']
    else:
        raise QuafuError(f'unsupported platform:{sys.platform}. '
                         f'You may raise a request issue on github.')
    return homedir

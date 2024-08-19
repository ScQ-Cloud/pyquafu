from typing import List
from quafu.transpile.passes.mapping.baselayout import Layout


class DataDict(dict):
    """A default dictionary-like object"""

    def __init__(self, *args, **kwargs):
        super(DataDict, self).__init__(*args, **kwargs)

        self['coupling_list']: List = None
        self['initial_layout']: Layout = None
        self['final_layout']: Layout = None
        self['variables']: List = None

    def __missing__(self, key):
        return None

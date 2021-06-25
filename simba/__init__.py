"""SIngle-cell eMBedding Along with features"""

from ._settings import settings
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from .readwrite import *
from . import datasets
from ._version import __version__


import sys
# needed when building doc (borrowed from scanpy)
sys.modules.update(
    {f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl']})

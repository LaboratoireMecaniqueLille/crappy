# coding: utf-8
import warnings

warnings.simplefilter("once", ImportWarning)

test = warnings.catch_warnings(record=True)
from . import technical
from . import sensor
from . import actuator
from . import blocks
from . import links
from ._deprecated import _deprecated
from .__version__ import __version__

print "TEST: ", test

#coding: utf-8
from __future__ import absolute_import

from .._global import NotInstalled
from .inout import InOut,MetaIO

from .agilent34420A import Agilent34420A
from .arduino import Arduino
try:
  from .comedi import Comedi
except ImportError:
  Comedi = NotInstalled('Comedi')
try:
  from .labjackT7 import Labjack_T7
except ImportError:
  Labjack_T7 = NotInstalled('Labjack_T7')
try:
  from .labjackUE9 import Labjack_UE9
except ImportError:
  Labjack_UE9 = NotInstalled('Labjack_UE9')
try:
  from .openDAQ import OpenDAQ
except ImportError:
  openDAQ = NotInstalled('OpenDAQ')

inout_list = MetaIO.IOclasses
in_list = MetaIO.Iclasses
in_list.update(inout_list)
out_list = MetaIO.Oclasses
out_list.update(inout_list)

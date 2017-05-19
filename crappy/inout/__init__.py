#coding: utf-8
from __future__ import absolute_import

from sys import platform
from .._global import NotInstalled,NotSupported
from .inout import InOut,MetaIO

from .agilent34420A import Agilent34420A
from .arduino import Arduino
from .opsens import Opsens
try:
  from .comedi import Comedi
except ImportError:
  Comedi = NotInstalled('Comedi')
try:
  from .labjackT7 import Labjack_t7
except ImportError:
  Labjack_t7 = NotInstalled('Labjack_t7')
try:
  from .labjackUE9 import Labjack_ue9
except ImportError:
  Labjack_ue9 = NotInstalled('Labjack_ue9')
try:
  from .opendaq import Opendaq
except ImportError:
  openDAQ = NotInstalled('OpenDAQ')

if 'win' in platform:
  try:
    from .daqmx import Daqmx
  except ImportError:
    Daqmx = NotInstalled('Daqmx')
else:
  Daqmx = NotSupported('Daqmx')

inout_list = MetaIO.IOclasses
in_list = MetaIO.Iclasses
in_list.update(inout_list)
out_list = MetaIO.Oclasses
out_list.update(inout_list)

#coding: utf-8
from __future__ import absolute_import

# On Linux, WindowsError does not exist, so to avoid NameError,
# make sure it is defined (and set it to None instead)
try:
  WindowsError
except NameError:
  WindowsError = None

from sys import platform
from .._global import NotInstalled,NotSupported
from .inout import InOut,MetaIO

from .agilent34420A import Agilent34420A
from .arduino import Arduino
from .opsens import Opsens

try:
  from .kollmorgen import Koll
except ImportError:
  Koll = NotInstalled("Koll")

try:
  from .spectrum import Spectrum
except (ImportError,OSError):
  Spectrum = NotInstalled("Spectrum")
try:
  from .comedi import Comedi
except (ImportError,WindowsError,OSError):
  Comedi = NotInstalled('Comedi')
try:
  from .labjackT7 import Labjack_t7
  from .t7Streamer import T7_streamer
except ImportError:
  Labjack_t7 = NotInstalled('Labjack_t7')
  T7_streamer = NotInstalled('T7_streamer')
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
  try:
    from .nidaqmx import Nidaqmx
  except ImportError:
    Nidaqmx = NotInstalled('Nidaqmx')
else:
  Daqmx = NotSupported('Daqmx')
  Nidaqmx = NotSupported('Nidaqmx')


inout_list = MetaIO.IOclasses
in_list = MetaIO.Iclasses
in_list.update(inout_list)
out_list = MetaIO.Oclasses
out_list.update(inout_list)

#coding: utf-8

from sys import platform
from .inout import InOut,MetaIO

from .agilent34420A import Agilent34420a
from .arduino import Arduino
from .comedi import Comedi
from .kollmorgen import Koll
from .labjackT7 import Labjack_t7
from .labjackUE9 import Labjack_ue9
from .opendaq import Opendaq
from .opsens import Opsens
from .spectrum import Spectrum
from .t7Streamer import T7_streamer

# Win specific
from .daqmx import Daqmx
from .nidaqmx import Nidaqmx

inout_list = MetaIO.IOclasses
in_list = MetaIO.Iclasses
in_list.update(inout_list)
out_list = MetaIO.Oclasses
out_list.update(inout_list)

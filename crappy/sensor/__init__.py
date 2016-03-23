# coding: utf-8
try:
	from ._ximeaSensor import Ximea
	import ximeaModule as ximeaModule
except:
    print "ximeaModule not installed"
try:
    from ._jaiSensor import Jai
    import clModule as clModule
except:
    print "Jai not compatible with this installation \n"
    
import platform as _platform
if(_platform.system()=="Linux"):
	from ._comediSensor import ComediSensor
	import comediModule as comediModule

from ._biotensSensor import BiotensSensor
from ._Agilent34420ASensor import Agilent34420ASensor

from ._dummySensor import DummySensor
from ._variateurTriboSensor import VariateurTriboSensor
from _lal300Sensor import SensorLal300
from . import clserial
try:
	from _niusb6008.py import DaqmxSensor
except:
	print "Cannot find Daqmx Drivers"

# coding: utf-8
#try:
#except:
    #print "ximeaModule not installed"
try:
    from ._jaiSensor import Jai
    import clModule as clModule
except:
    print "Jai not compatible with this installation \n"
# try:
import ximeaModule as ximeaModule
from ._ximeaSensor import Ximea
# except:
# 	print "Cannot load ximea Module"
    
import platform as _platform
if(_platform.system()=="Linux"):
	from ._comediSensor import ComediSensor
	import comediModule as comediModule

from ._biotensSensor import BiotensSensor
from ._Agilent34420ASensor import Agilent34420ASensor

from ._dummySensor import DummySensor
from ._variateurTriboSensor import VariateurTriboSensor
from _lal300Sensor import SensorLal300
try:
	from _niusb6008 import DaqmxSensor
except:
	print "cannot find Daqmx Drivers \n"
from . import clserial
# coding: utf-8
#try:
#except:
    #print "ximeaModule not installed"
try:
    from ._jaiSensor import Jai
    import clModule as clModule
except Exception as e:
    print "WARNING: ", e
                
try:
    import ximeaModule as ximeaModule
    from ._ximeaSensor import Ximea
except Exception as e:
    print "WARNING: ", e

try:
    import pyFgenModule as pyFgen 
except Exception as e:
    print "WARNING: ", e
     
import platform as _platform
if(_platform.system()=="Linux"):
    try:
        from ._comediSensor import ComediSensor
        import comediModule as comediModule
    except Exception as e:
        print "WARNING: ", e
                
from ._biotensSensor import BiotensSensor
from ._Agilent34420ASensor import Agilent34420ASensor

from ._dummySensor import DummySensor
from ._variateurTriboSensor import VariateurTriboSensor
from _lal300Sensor import SensorLal300
try:
	from _niusb6008 import DaqmxSensor
except Exception as e:
    print "WARNING: ", e

from . import clserial
try:
    from _labJackSensor import LabJackSensor
except Exception as e:
    print "WARNING: ", e

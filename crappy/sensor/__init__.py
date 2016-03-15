# coding: utf-8
#from ._ximeaSensor import XimeaSensor
try:
    from ._ximeaSensor import Ximea
except:
    print "ximeaModule not installed"
try:
    from ._jaiSensor import Jai
except:
    print "Jai not compatible with this installation \n"
    
from ._comediSensor import ComediSensor
from ._biotensSensor import BiotensSensor
from ._Agilent34420ASensor import Agilent34420ASensor
import comediModule as comediModule
import clModule as clModule
#import ximeaModule as ximeaModule

from ._dummySensor import DummySensor
from ._variateurTriboSensor import VariateurTriboSensor
from _lal300Sensor import SensorLal300
from . import clserial

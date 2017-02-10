#!/usr/bin/python
# coding: utf-8

##  @defgroup sensor Sensor
# The sensors represent everything that can acquire a physical signal. It can be an acquisition card,
# but also a camera, a thermocouple...
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Import classes to put them in the current namespace.
#
# @author Robin Siemiatkowski, Victor Couty
# @version 0.1
# @date 21/06/2016

from os import popen as _popen
import platform as _platform
from .._warnings import import_error
from _biotensSensor import BiotensSensor
from _biaxeSensor import BiaxeSensor
from _Agilent34420ASensor import Agilent34420ASensor
from _CMdriveSensor import CmDriveSensor
from _dummySensor import DummySensor
from _variateurTriboSensor import VariateurTriboSensor
from _lal300Sensor import Lal300Sensor, SensorLal300
from _PISensor import PISensor

#Cameras
try:
  import ximeaModule as ximeaModule
  from _ximea import Ximea
except Exception as e:
  import_error(e.message)
from _webcam import Webcam
from _fakeCameraSensor import Fake_camera
try:
  from _ximeaCV import XimeaCV
except Exception as e:
  import_error(e.message)

if _platform.system() == "Linux":
  try:
    from _comediSensor import ComediSensor
    import comediModule as comediModule
  except Exception as e:
    import_error(e.message)

  _p = _popen("lsmod |grep menable")
  if len(_p.read()) != 0:
    try:
      import clModule as clModule
      from _jaiSensor import Jai
      from _clserial import _clSerial
      from _clserial import _jaiSerial

      JaiSerial = _jaiSerial.JaiSerial
      ClSerial = _clSerial.ClSerial
    except Exception as e:
      import_error(e.message)

if _platform.system() == "Windows":
  try:
    import pyFgenModule as pyFgen
  except Exception as e:
    import_error(e.message)

  if len(_popen('driverquery /NH |findstr "me4"').read()) != 0:
    try:
      import clModule as clModule
      from _jaiSensor import Jai

      from _clserial import _clSerial
      from _clserial import _jaiSerial

      JaiSerial = _jaiSerial.JaiSerial
      ClSerial = _clSerial.ClSerial
    except Exception as e:
      import_error(e.message)

try:
  from _orientalSensor import OrientalSensor
except Exception as e:
  import_error(e.message)

try:
  from _daqmxSensor import DaqmxSensor
except Exception as e:
  import_error(e.message)

try:
  from _labJackSensor import LabJackSensor
except Exception as e:
  import_error(e.message)

try:
  del e
except NameError:
  pass
del _popen, _platform, import_error

# @}
# @}

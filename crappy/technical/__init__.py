#!/usr/bin/python
# coding: utf-8

##  @defgroup technical Technical
# @brief Some hardware is both a sensor and an actuator by our definitions.
# This is for example the case of a variable-frequency drive : they can set the speed of the motor (the actuator part),
# but most of them can also read the position or the speed of the motor the sensor part).
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Import classes to put them in the current namespace.
#
# @author Robin Siemiatkowski, Victor Couty
# @version 0.2
# @date 24/01/2017

from .._warnings import import_error

__motors__ = ['Biotens', 'Biaxe', 'Lal300', 'PI', 'VariateurTribo', 'CmDrive', 'Oriental', 'DummyTechnical']
__boardnames__ = ['Comedi', 'Daqmx', 'LabJack', 'Agilent34420A', 'fgen', 'DummySensor']

from _biaxeTechnical import Biaxe
from _biotensTechnical import Biotens
from _lal300Technical import Lal300
from _PITechnical import PI
from _dummyTechnical import DummyTechnical
from _variateurTribo import VariateurTribo
from _acquisition import Acquisition
from _command import Command
from _motion import Motion
from _CMdriveTechnical import CmDrive
from _interfaceCMdrive import Interface
from _orientalTechnical import Oriental
from _conditionneur_5018 import Conditionner_5018
from _datapicker import DataPicker
from _cameraConfig import camera_config

try:
  from _labjack import LabJack
except Exception as e:
  print e

try:
  from _OpenDAQ import OpenDAQ
except Exception as e:
  print e

try:
  from _cameraInit import get_camera_config
except Exception as e:
  import_error(e.message)
try:
  from _technicalCamera import TechnicalCamera
except Exception as e:
  import_error(e.message)
try:
  from _correl import TechCorrel
except Exception as e:
  import_error(e.message)

del e, import_error

# Uncomment the following lines for testing please see the documentation ("how to bind C/C++ with Python")
# try:
# import helloModule as helloModule
# except Exception as e:
# print e

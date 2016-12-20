##  @defgroup actuator Actuator
# The actuators represent all the objects that can interact on the other part of the test, and can be controled.
# The most common example are motors.
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Import classes to put them in the current namespace.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

import platform
from .._warnings import import_error

from ._meta import command, motion

e = None
if platform.system() == "Linux":
  try:
    from ._comediActuator import ComediActuator
  except Exception as e:
    import_error(e.message)

from _biaxeActuator import BiaxeActuator
from _biotensActuator import BiotensActuator
from _PIActuator import PIActuator
from _CMdriveActuator import CmDriveActuator
from _dummyActuator import DummyActuator

try:
  from ._labJackActuator import LabJackActuator
except Exception as e:
  import_error(e.message)
from _lal300Actuator import Lal300Actuator, ActuatorLal300

from _variateurTriboActuator import VariateurTriboActuator

try:
  from _daqmxActuator import DaqmxActuator
except Exception as e:
  import_error(e.message)
try:
  from _orientalActuator import OrientalActuator
except Exception as e:
  import_error(e.message)

del platform, e, import_error

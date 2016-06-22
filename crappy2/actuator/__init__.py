import platform
import warnings

# warnings.simplefilter("once", ImportWarning)
e = None
if platform.system() == "Linux":
    try:
        from ._comediActuator import ComediActuator
    except Exception as e:
        warnings.warn(e.message, ImportWarning)

from ._biaxeActuator import BiaxeActuator
from ._biotensActuator import BiotensActuator
from ._PIActuator import PIActuator
from ._CMdriveActuator import CmDriveActuator

try:
    from ._labJackActuator import LabJackActuator
except Exception as e:
    warnings.warn(e.message, ImportWarning)
    pass
from _lal300Actuator import Lal300Actuator, ActuatorLal300

from ._variateurTriboActuator import VariateurTriboActuator

try:
    from _daqmxActuator import DaqmxActuator
except Exception as e:
    warnings.warn(e.message, ImportWarning)

from ._meta import command, motion

del platform, e

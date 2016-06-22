# coding: utf-8
import warnings
e = None
# warnings.simplefilter("once", ImportWarning)
try:
    from ._biotensTechnical import Biotens
except Exception as e:
    warnings.warn(e.message, ImportWarning)

from ._biaxeTechnical import Biaxe
from ._biotensTechnical import Biotens
from ._lal300Technical import Lal300
from ._PITechnical import PI

__motors__ = ['Biotens', 'Biaxe', 'Lal300', 'PI', 'VariateurTribo', 'CmDrive']
__boardnames__ = ['Comedi', 'Daqmx', 'LabJack', 'Agilent34420A', 'fgen']
__cameras__ = ['Ximea', 'Jai']

from ._variateurTribo import VariateurTribo
from ._acquisition import Acquisition
from ._command import Command
from ._motion import Motion
from ._CMdriveTechnical import CmDrive
from ._interfaceCMdrive import Interface
# from ._jaiTechnical import Jai
from ._cameraInit import get_camera_config
from ._technicalCamera import TechnicalCamera

# from . import *
# __all__ = ['Ximea']
try:
    from ._correl import TechCorrel
except Exception as e:
    warnings.warn(e.message, ImportWarning)

del e, warnings

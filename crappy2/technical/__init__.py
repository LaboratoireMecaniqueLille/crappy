# coding: utf-8
from .._warnings import import_error

e = None
try:
    from ._biotensTechnical import Biotens
except Exception as e:
    import_error(e.message)

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
try:
    from ._cameraInit import get_camera_config
except Exception as e:
    import_error(e.message)
try:
    from ._technicalCamera import TechnicalCamera
except Exception as e:
    import_error(e.message)
# from . import *
# __all__ = ['Ximea']
try:
    from ._correl import TechCorrel
except Exception as e:
    import_error(e.message)
from ._datapicker import DataPicker
del e, import_error

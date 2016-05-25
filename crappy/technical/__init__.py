# coding: utf-8
#def main():
    #"""Entry point for the application script"""
    #print("Call your main application code here")

try:
    from ._biotensTechnical import Biotens
except Exception as _e:
    print 'WARNING: ', _e
    del(_e)

from ._biaxeTechnical import Biaxe
from ._biotensTechnical import Biotens
from ._lal300Technical import Lal300
from ._PITechnical import PI

__motors__ = ['Biotens', 'Biaxe', 'Lal300', 'PI', 'VariateurTribo']
__boardnames__ = ['Comedi', 'Daqmx', 'LabJack', 'Agilent34420A', 'fgen']
__cameras__ = ['Ximea', 'Jai']

from ._variateurTribo import VariateurTribo
from ._acquisition import Acquisition
from ._command import Command
from ._motion import Motion
from ._CMdriveTechnical import CmDrive
from ._interfaceCMdrive import Interface
#from ._jaiTechnical import Jai
from ._cameraInit import getCameraConfig
from ._technicalCamera import TechnicalCamera
#from . import *
#__all__ = ['Ximea']

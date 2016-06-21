# coding: utf-8
# def main():
# """Entry point for the application script"""
# print("Call your main application code here")

# from . import technical
import platform

if platform.system() == "Linux":
    from ._commandComedi import CommandComedi
    from ._measureComediByStep import MeasureComediByStep
    from ._streamerComedi import StreamerComedi

from ._cameraDisplayer import CameraDisplayer
from ._commandBiaxe import CommandBiaxe
from ._commandBiotens import CommandBiotens
from _autoDrive import AutoDrive
from ._commandPI import CommandPI
from ._compacter import Compacter
from ._grapher import Grapher
from ._measureAgilent34420A import MeasureAgilent34420A
from ._measureByStep import MeasureByStep
from ._multiPath import MultiPath
from ._pid import PID
from ._reader import Reader
from ._saver import Saver
from ._server import Server
from ._client import Client
# from ._signalAdapter import SignalAdapter
from ._signalGenerator import SignalGenerator

try:
    from ._streamerCamera import StreamerCamera
except Exception as e:
    print "WARNING: ", e

from ._streamer import Streamer

try:
    from ._videoExtenso import VideoExtenso
except Exception as e:
    print "WARNING: ", e

from _interfaceTribo import Interface
from _lal300Command import CommandLal300
from ._meta import MasterBlock

del platform

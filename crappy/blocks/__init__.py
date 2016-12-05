# coding: utf-8
##  @defgroup blocks Blocks
# Crappy is based on a schematic architecture with single blocks linked between each others.
# blocks are the part that do and the Links are the parts that carry informations between these blocks.
#
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual), but you can also implement your own.
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Import classes to put them in the current namespace.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 05/07/2016

import platform
from .._warnings import import_error

e = None

if platform.system() == "Linux":
    from ._commandComedi import CommandComedi
    from ._measureComediByStep import MeasureComediByStep
    from ._streamerComedi import StreamerComedi

from ._cameraDisplayer import CameraDisplayer
from ._commandBiaxe import CommandBiaxe
from ._commandBiotens import CommandBiotens
from ._commandBiotens_v2 import CommandBiotens_v2
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
from ._canvasdrawing import CanvasDrawing
from ._fakeCamera import FakeCamera
from ._sink import Sink

try:
    from ._streamerCamera import StreamerCamera
except Exception as e:
    import_error(e.message)

from ._streamer import Streamer

try:
    from ._videoExtenso import VideoExtenso
except Exception as e:
    import_error(e.message)

from _sendPath import InterfaceSendPath
from _saverTriggered import SaverTriggered
from _interfaceTribo import InterfaceTribo
from _tribo_manual_interface import InterfaceManual
from _lal300Command import CommandLal300

try:
    from ._correl import Correl
except Exception as e:
    import_error(e.message)

from ._meta import MasterBlock
from ._commandCegitab import SerialPortActuator, SerialPortCaptor, PipeCegitab
from ._savergui import SaverGUI
from ._gui import InterfaceTomo4D

del e, platform, import_error

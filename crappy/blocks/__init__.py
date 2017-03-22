# coding: utf-8
##  @defgroup blocks Blocks
# Crappy is based on a schematic architecture with single blocks linked between each others.
# blocks are the part that do and the Links are the parts that carry informations between these blocks.
#
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Import classes to put them in the current namespace.
#
# @author Victor Couty, Robin Siemiatkowski
# @version 0.1
# @date 05/07/2016

import platform

from .._global import NotInstalled,NotSupported
from .masterblock import MasterBlock

if platform.system() == "Linux":
  from .commandComedi import CommandComedi
else:
  CommandComedi = NotSupported('CommandComedi')

from .displayer import Displayer
from .commandBiaxe import CommandBiaxe
from .commandBiotens import CommandBiotens
from .autoDrive import AutoDrive
from .commandPI import CommandPI
from .grapher import Grapher
from .measureAgilent34420A import MeasureAgilent34420A
from .measureByStep import MeasureByStep
from .multiPath import MultiPath
from .reader import Reader
from .saver import Saver
from .server import Server
from .client import Client
from .signalGenerator import SignalGenerator
from .canvasdrawing import CanvasDrawing
from .fakeCamera import FakeCamera
from .sink import Sink
#from .videoExtenso import VideoExtenso ## TO FIX
from .camera import Camera
from .streamer import Streamer
from .dashboard import Dashboard
from .controlcommand import ControlCommand
from .wavegenerator import WaveGenerator
from .dataReader import DataReader
from .saverTriggered import SaverTriggered
from .interfaceTribo import InterfaceTribo
from .lal300Command import CommandLal300
from .commandCegitab import PipeCegitab
from .savergui import SaverGUI
from .gui import InterfaceTomo4D
from .pidtomo import PIDTomo

try:
  from .correl import Correl
except ImportError:
  Correl = NotInstalled('Correl')

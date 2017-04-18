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
# @author Victor Couty
# @version 0.1
# @date 13/04/2017

from .._global import NotInstalled,NotSupported
from .autoDrive import AutoDrive
from .camera import Camera
from .dashboard import Dashboard
from .displayer import Displayer
from .grapher import Grapher
from .ioblock import IOBlock
from .masterblock import MasterBlock
from .measureByStep import MeasureByStep
from .reader import Reader
from .saver import Saver
from .signalGenerator import SignalGenerator
from .sink import Sink
from .videoExtenso import Video_extenso
from .wavegenerator import WaveGenerator

try:
  from .correl import Correl
except ImportError:
  Correl = NotInstalled('Correl')

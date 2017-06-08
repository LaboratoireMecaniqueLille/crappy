# coding: utf-8
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.

from .._global import NotInstalled,NotSupported
from .autoDrive import AutoDrive
from .camera import Camera
from .dashboard import Dashboard
from .displayer import Displayer
from .generator import Generator
from .grapher import Grapher
from .ioblock import IOBlock
from .machine import Machine
from .masterblock import MasterBlock
from .measureByStep import MeasureByStep
from .multiplex import Multiplex
from .reader import Reader
from .saver import Saver
from .signalGenerator import SignalGenerator
from .sink import Sink
from .videoExtenso import Video_extenso

try:
  from .correl import Correl
except ImportError:
  Correl = NotInstalled('Correl')

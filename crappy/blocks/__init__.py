# coding: utf-8
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.

from .._global import NotInstalled,NotSupported
from .autoDrive import AutoDrive
from .camera import Camera
from .client import Client
from .dashboard import Dashboard
from .displayer import Displayer
from .drawing import Drawing
from .generator import Generator
from .gui import GUI
from .grapher import Grapher
from .ioblock import IOBlock
from .machine import Machine
from .mean import Mean
from .masterblock import MasterBlock
from .multiplex import Multiplex
from .pid import PID
from .reader import Reader
from .saver import Saver
from .server import Server
from .sink import Sink
from .videoExtenso import Video_extenso

try:
  from .gpucorrel import GPUCorrel
except ImportError:
  GPUCorrel = NotInstalled('GPUCorrel')

try:
  from .discorrel import DISCorrel
except ImportError:
  DISCorrel = NotInstalled('DISCorrel')

try:
  from .hdf_saver import Hdf_saver
except ImportError:
  Hdf_saver = NotInstalled("Hdf_saver")

try:
  from .signalGenerator import SignalGenerator
except ImportError:
  SignalGenerator = NotInstalled('SignalGenerator')

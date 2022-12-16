# coding: utf-8
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.

from .autoDrive import AutoDrive
from .block import Block
from .camera import Camera
from .client_server import Client_server
from .dashboard import Dashboard
from .discorrel import DISCorrel
from .displayer import Displayer
from .disve import DISVE
from .drawing import Drawing
from .fake_machine import Fake_machine
from .generator import Generator
from .gpucorrel import GPUCorrel
from .gpuve import GPUVE
from .grapher import Grapher
from .gui import GUI
from .hdf_recorder import Hdf_recorder, Hdf_saver
from .ioblock import IOBlock
from .machine import Machine
from .mean import Mean_block
from .multiplex import Multiplex
from .pid import PID
from .reader import Reader
from .recorder import Recorder, Saver
from .sink import Sink
from .ucontroller import UController
from .videoExtenso import Video_extenso

from .camera_parallel import Camera_parallel
from .videoExtenso_parallel import Video_extenso_parallel
from .disve_parallel import Disve_parallel
from .discorrel_parallel import Discorrel_parallel
from .gpuve_parallel import Gpuve_parallel

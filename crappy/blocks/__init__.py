# coding: utf-8
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.

from .autoDrive import AutoDrive
from .meta_block import Block
from .camera import Camera
from .client_server import Client_server
from .dashboard import Dashboard
from .discorrel import Discorrel
from .disve import Disve
from .drawing import Drawing
from .fake_machine import Fake_machine
from .generator import Generator
from .gpu_correl import GpuCorrel
from .gpu_ve import GpuVe
from .grapher import Grapher
from .gui import GUI
from .hdf_recorder import Hdf_recorder
from .ioblock import IOBlock
from .machine import Machine
from .mean import Mean_block
from .multiplex import Multiplex
from .pid import PID
from .reader import Reader
from .recorder import Recorder
from .sink import Sink
from .ucontroller import UController
from .video_extenso import VideoExtenso

from . import generator_path
from . import camera_processes

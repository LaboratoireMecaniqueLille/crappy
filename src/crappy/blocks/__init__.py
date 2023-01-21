# coding: utf-8
# Blocks are classes, running indefinitely in a single process.
# Some of them are already implemented (see the reference manual),
# but you can also implement your own.

from .auto_drive import AutoDrive
from .meta_block import Block
from .camera import Camera
from .client_server import ClientServer
from .dashboard import Dashboard
from .dis_correl import DISCorrel
from .dic_ve import DICVE
from .drawing import Drawing
from .fake_machine import FakeMachine
from .generator import Generator
from .gpu_correl import GPUCorrel
from .gpu_ve import GPUVE
from .grapher import Grapher
from .gui import GUI
from .hdf_recorder import HDFRecorder
from .ioblock import IOBlock
from .machine import Machine
from .mean import MeanBlock
from .multiplex import Multiplex
from .pid import PID
from .reader import Reader
from .recorder import Recorder
from .sink import Sink
from .ucontroller import UController
from .video_extenso import VideoExtenso

from . import generator_path
from . import camera_processes

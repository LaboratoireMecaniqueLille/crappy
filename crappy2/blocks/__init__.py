# coding: utf-8
import platform
import warnings

e = None

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
    warnings.warn(e.message, ImportWarning)

from ._streamer import Streamer

try:
    from ._videoExtenso import VideoExtenso
except Exception as e:
    warnings.warn(e.message, ImportWarning)

from _interfaceTribo import Interface
from _lal300Command import CommandLal300

try:
    from ._correl import Correl
except Exception as e:
    warnings.warn(e.message, ImportWarning)

from ._meta import MasterBlock

del e, platform

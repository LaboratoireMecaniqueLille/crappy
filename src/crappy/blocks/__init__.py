# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from importlib.util import resolve_name as _resolve_name
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'AutoDriveVideoExtenso': ('.auto_drive_video_extenso',
                                      'AutoDriveVideoExtenso'),
            'Button': ('.button', 'Button'),
            'Camera': ('.camera', 'Camera'),
            'Canvas': ('.canvas', 'Canvas'),
            'ClientServer': ('.client_server', 'ClientServer'),
            'Dashboard': ('.dashboard', 'Dashboard'),
            'DICVE': ('.dic_ve', 'DICVE'),
            'DISCorrel': ('.dis_correl', 'DISCorrel'),
            'FakeMachine': ('.fake_machine', 'FakeMachine'),
            'Generator': ('.generator', 'Generator'),
            'GPUCorrel': ('.gpu_correl', 'GPUCorrel'),
            'GPUVE': ('.gpu_ve', 'GPUVE'),
            'Grapher': ('.grapher', 'Grapher'),
            'HDFRecorder': ('.hdf_recorder', 'HDFRecorder'),
            'IOBlock': ('.ioblock', 'IOBlock'),
            'LinkReader': ('.link_reader', 'LinkReader'),
            'Machine': ('.machine', 'Machine'),
            'MeanBlock': ('.mean', 'MeanBlock'),
            'Multiplexer': ('.multiplexer', 'Multiplexer'),
            'Pause': ('.pause', 'Pause'),
            'PID': ('.pid', 'PID'),
            'Recorder': ('.recorder', 'Recorder'),
            'Scheduler': ('.scheduler', 'Scheduler'),
            'Sink': ('.sink', 'Sink'),
            'StopBlock': ('.stop_block', 'StopBlock'),
            'StopButton': ('.stop_button', 'StopButton'),
            'Synchronizer': ('.synchronizer', 'Synchronizer'),
            'UController': ('.ucontroller', 'UController'),
            'VideoExtenso': ('.video_extenso', 'VideoExtenso'),
            'generator_path': ('.generator_path', None),
            'schedulers': ('.schedulers', None),
            'camera_processes': ('.camera_processes', None),
            'vision': ('.vision', None)}


def is_reserved(cls: type) -> bool:
  """Reserves Block names without loading their implementations.

  .. versionadded:: 2.2.0
  """

  from .vision import _EXPORTS as vision_exports

  for exports, package in ((_EXPORTS, __name__),
                           (vision_exports, __name__ + '.vision')):
    reference = exports.get(cls.__name__)
    if reference is not None and reference[1] is not None:
      return cls.__module__ != _resolve_name(reference[0], package)
  return False


from .meta_block import Block

__all__ = ('Block', *tuple(_EXPORTS))

if _TYPE_CHECKING:
  from .auto_drive_video_extenso import AutoDriveVideoExtenso
  from .button import Button as Button
  from .camera import Camera as Camera
  from .canvas import Canvas as Canvas
  from .client_server import ClientServer as ClientServer
  from .dashboard import Dashboard as Dashboard
  from .dic_ve import DICVE as DICVE
  from .dis_correl import DISCorrel as DISCorrel
  from .fake_machine import FakeMachine as FakeMachine
  from .generator import Generator as Generator
  from .gpu_correl import GPUCorrel as GPUCorrel
  from .gpu_ve import GPUVE as GPUVE
  from .grapher import Grapher as Grapher
  from .hdf_recorder import HDFRecorder as HDFRecorder
  from .ioblock import IOBlock as IOBlock
  from .link_reader import LinkReader as LinkReader
  from .machine import Machine as Machine
  from .mean import MeanBlock as MeanBlock
  from .multiplexer import Multiplexer as Multiplexer
  from .pause import Pause as Pause
  from .pid import PID as PID
  from .recorder import Recorder as Recorder
  from .scheduler import Scheduler as Scheduler
  from .sink import Sink as Sink
  from .stop_block import StopBlock as StopBlock
  from .stop_button import StopButton as StopButton
  from .synchronizer import Synchronizer as Synchronizer
  from .ucontroller import UController as UController
  from .video_extenso import VideoExtenso as VideoExtenso
  from . import generator_path as generator_path
  from . import schedulers as schedulers
  from . import camera_processes as camera_processes
  from . import vision as vision


def __getattr__(name: str) -> _Any:
  """Loads and caches one public export on first access.

  .. versionadded:: 2.2.0
  """

  if name not in _EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

  module, attribute = _EXPORTS[name]
  imported = _import_module(module, __name__)
  value = imported if attribute is None else getattr(imported, attribute)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))

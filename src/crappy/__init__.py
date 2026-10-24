# coding: utf-8

"""This file is the entry point to the module Crappy.

It imports the runtime modules and defines aliases. Bundled image resources
are loaded on first access.
"""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

# The modules that are always eagerly imported
from . import blocks
from . import links

# Importing other features
from .__version__ import __version__
from ._global import OptionalModule, docs

if _TYPE_CHECKING:
  from . import actuator, camera, inout, lamcube, modifier, tool
  from .actuator import Actuator
  from .camera import Camera
  from .inout import InOut
  from .modifier import Modifier
  from .blocks.vision import VisionBlock
  from .blocks.generator_path.meta_path import Path
  from ._resources import resources

_EXPORTS = {'actuator': ('.actuator', None),
            'Actuator': ('.actuator', 'Actuator'),
            'camera': ('.camera', None),
            'Camera': ('.camera', 'Camera'),
            'inout': ('.inout', None),
            'InOut': ('.inout', 'InOut'),
            'lamcube': ('.lamcube', None),
            'modifier': ('.modifier', None),
            'Modifier': ('.modifier', 'Modifier'),
            'Path': ('.blocks.generator_path.meta_path', 'Path'),
            'tool': ('.tool', None),
            'VisionBlock': ('.blocks.vision', 'VisionBlock'),
            'resources': ('._global', 'resources')}

__all__ = ('actuator', 'Actuator',
           'blocks', 'Block', 'VisionBlock', 'Path',
           'camera', 'Camera',
           'inout', 'InOut',
           'lamcube',
           'links', 'link', 'img_link', 'display_graph',
           'modifier', 'Modifier',
           'tool',
           'docs',
           'OptionalModule', 'resources',
           'stop', 'prepare', 'launch', 'start', 'renice', 'reset',
           '__version__')

# Useful aliases
link = links.link
img_link = links.img_link
display_graph = links.link_graph.display
Block = blocks.Block

# Useful commands
stop = Block.stop_all
prepare = Block.prepare_all
launch = Block.launch_all
start = Block.start_all
renice = Block.renice_all
reset = Block.reset


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
  """Includes lazy exports without loading them.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))

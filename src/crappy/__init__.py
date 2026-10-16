# coding: utf-8

"""This file is the entry point to the module Crappy.

It imports the runtime modules and defines aliases. Bundled image resources
are loaded on first access.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

# Importing the modules of crappy
from . import actuator
from . import blocks
from . import camera
from . import inout
from . import lamcube
from . import links
from . import modifier
from . import tool

# Importing other features
from .__version__ import __version__
from ._global import OptionalModule, docs

if _TYPE_CHECKING:
  from ._resources import resources

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
VisionBlock = blocks.vision.VisionBlock
Actuator = actuator.Actuator
Camera = camera.Camera
InOut = inout.InOut
Modifier = modifier.Modifier
Path = blocks.generator_path.meta_path.Path

# Useful commands
stop = Block.stop_all
prepare = Block.prepare_all
launch = Block.launch_all
start = Block.start_all
renice = Block.renice_all
reset = Block.reset


def __getattr__(name: str) -> type:
  """Loads the resource namespace on first access."""

  if name == 'resources':
    from ._global import resources
    globals()[name] = resources
    return resources

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  """Includes lazy exports without loading them."""

  return sorted(set(globals()) | set(__all__))

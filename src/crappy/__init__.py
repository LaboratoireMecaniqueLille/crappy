# coding: utf-8

"""This file is the entry point to the module Crappy.

It imports all the modules and other resources, and defines aliases.
"""

from pkg_resources import resource_string, resource_filename
from numpy import frombuffer, uint8
from webbrowser import open

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
from ._global import OptionalModule

# Useful aliases
link = links.link
Block = blocks.Block
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


# Quick access to documentation
def docs():
  """Opens the online documentation of Crappy.

  It opens the latest version, and of course requires an internet access.
  
  .. versionadded:: 1.5.5
  .. versionchanged:: 2.0.0 renamed from doc to docs
  """

  open('https://crappy.readthedocs.io/en/latest/')


# Data aliases
class resources:
  """This class defines aliases for quick access to the resources in the
  `tool/data/` folder.

  These aliases are then used in the examples provided on the GitHub
  repository, but could also be used in custom user scripts.

  .. versionadded:: 1.5.3
  """

  try:
    # Defining aliases to the images themselves
    from cv2 import imdecode
    speckle = imdecode(frombuffer(resource_string('crappy',
                                                  'tool/data/speckle.png'),
                                  uint8), flags=0)

    ve_markers = imdecode(frombuffer(
      resource_string('crappy', 'tool/data/ve_markers.tif'), uint8), flags=0)

    pad = imdecode(frombuffer(resource_string('crappy', 'tool/data/pad.png'),
                              uint8), flags=0)

  # In case the module opencv-python is missing
  except (ModuleNotFoundError, ImportError):
    speckle = OptionalModule('opencv-python')
    ve_markers = OptionalModule('opencv-python')
    pad = OptionalModule('opencv-python')

  # Also getting the paths to the images
  paths = {'pad': resource_filename('crappy', 'tool/data/pad.png'),
           'speckle': resource_filename('crappy', 'tool/data/speckle.png'),
           've_markers': resource_filename('crappy',
                                           'tool/data/ve_markers.tif')}

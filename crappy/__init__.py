# coding: utf-8

from pkg_resources import resource_string, resource_filename
from numpy import frombuffer, uint8
from ._global import OptionalModule

from . import actuator
from . import camera
from . import inout
from . import tool
from . import blocks
from . import links
from . import modifier
from .__version__ import __version__

# For compatibility (deprecated!)
condition = modifier

# Useful aliases
link = links.link
Block = blocks.Block
stop = Block.stop_all
prepare = Block.prepare_all
launch = Block.launch_all
start = Block.start_all
renice = Block.renice_all
reset = Block.reset


# Data aliases
class resources:
  try:
    from cv2 import imdecode
    speckle = imdecode(frombuffer(resource_string('crappy',
                                                  'tool/data/speckle.png'),
                                  uint8), flags=0)

    ve_markers = imdecode(frombuffer(
      resource_string('crappy', 'tool/data/ve_markers.tif'), uint8), flags=0)

    pad = imdecode(frombuffer(resource_string('crappy', 'tool/data/pad.png'),
                              uint8), flags=0)
  except (ModuleNotFoundError, ImportError):
    speckle = OptionalModule('opencv-python')
    ve_markers = OptionalModule('opencv-python')
    pad = OptionalModule('opencv-python')

  paths = {'pad': resource_filename('crappy', 'tool/data/pad.png'),
           'speckle': resource_filename('crappy', 'tool/data/speckle.png'),
           've_markers': resource_filename('crappy',
                                           'tool/data/ve_markers.tif')}

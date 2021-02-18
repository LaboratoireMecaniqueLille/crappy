# coding: utf-8

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

link = links.link
Block = blocks.Block
stop = Block.stop_all
prepare = Block.prepare_all
launch = Block.launch_all
start = Block.start_all
renice = Block.renice_all
reset = Block.reset

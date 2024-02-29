# coding: utf-8

"""This file is the entry point to the module Crappy.

It imports all the modules and other resources, and defines aliases.
"""

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
from ._global import OptionalModule, docs, resources

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

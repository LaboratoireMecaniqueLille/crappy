# coding: utf-8

import actuator
import camera
#import inout
import tool
import blocks
import links
from .__version__ import __version__

link = links.link
MasterBlock = blocks.MasterBlock
stop = MasterBlock.stop_all
prepare = MasterBlock.prepare
launch = MasterBlock.launch
start = MasterBlock.start_all

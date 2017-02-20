# coding: utf-8

import sensor
import actuator
import technical
import blocks
import links

link = links.link
from _warnings import deprecated, import_error
from __version__ import __version__
#from _stop import stop
from blocks import MasterBlock
#start = MasterBlock.start_all
stop = MasterBlock.stop_all
prepare = MasterBlock.prepare
launch = MasterBlock.launch
start = MasterBlock.start_all

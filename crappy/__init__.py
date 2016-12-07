# coding: utf-8
from __future__ import print_function

import time
from . import sensor
from . import actuator
from . import technical
from . import blocks
from . import links
from ._warnings import deprecated, import_error
from .__version__ import __version__

def start(verbose=True):
  if verbose:
    def vprint(*args):
      print("[crappy.start]",*args)
  else:
    vprint = lambda x: None
  t0=time.time()
  vprint("Setting t0 to",time.strftime("%d %b %Y, %H:%M:%S"
                                      ,time.localtime(t0)))
  for instance in blocks._meta.MasterBlock.instances:
    instance.t0 = t0

  vprint("Starting the blocks...")
  for instance in blocks._meta.MasterBlock.instances:
    vprint("Starting",instance)
    instance.start()
    vprint("Started, PID:",instance.pid)
  vprint("All blocks are started.")

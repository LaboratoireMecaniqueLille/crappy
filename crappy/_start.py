# coding: utf-8
from __future__ import print_function

import time
from blocks._masterblock import MasterBlock

def start(verbose=True):
  if verbose:
    def vprint(*args):
      print("[crappy.start]",*args)
  else:
    vprint = lambda *x: None
  t0=time.time()
  vprint("Setting t0 to",time.strftime("%d %b %Y, %H:%M:%S"
                                      ,time.localtime(t0)))
  for instance in MasterBlock.instances:
    instance.t0 = t0

  vprint("Starting the blocks...")
  try:
    for instance in MasterBlock.instances:
      vprint("Starting",instance)
      instance.start()
      vprint("Started, PID:",instance.pid)
    vprint("All blocks are started.")
  except (Exception, KeyboardInterrupt) as e:
    print("Exception in main :", e)
    for instance in MasterBlock.instances:
      try:
        instance.stop()
      except Exception as e:
        print(e)
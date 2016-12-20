# coding: utf-8
from __future__ import print_function

from blocks._masterblock import MasterBlock


def stop(verbose=True):
  if verbose:
    def vprint(*args):
      print("[crappy.stop]", *args)
  else:
    vprint = lambda *x: None
  vprint("Stopping the blocks...")
  try:
    for instance in MasterBlock.instances:
      vprint("Stopping", instance, "(PID:{}".format(instance.pid))
      instance.stop()
      vprint("Stopped")
    vprint("All blocks are stopped.")
  except (Exception, KeyboardInterrupt) as e:
    print("Exception in main :", e)
    for instance in MasterBlock.instances:
      try:
        instance.stop()
      except Exception as e:
        print(e)

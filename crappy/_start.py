# coding: utf-8
from __future__ import print_function

import time
from blocks import MasterBlock


def launch(t0=None, verbose=True):
  if verbose:
    def vprint(*args):
      print("[crappy.launch]", *args)
  else:
    vprint = lambda *x: None
  if not t0:
    t0 = time.time()
  vprint("Setting t0 to", time.strftime("%d %b %Y, %H:%M:%S"
                                        , time.localtime(t0)))
  for instance in MasterBlock.instances:
    instance.launch(t0)
  t1 = time.time()
  vprint("All blocks loop started. It took", (t1 - t0) * 1000, "ms")


def start(verbose=True, start_loop=True):
  if verbose:
    def vprint(*args):
      print("[crappy.start]", *args)
  else:
    vprint = lambda *x: None

  vprint("Starting the blocks...")
  try:
    for instance in MasterBlock.instances:
      if instance.status == "idle":
        vprint("Starting", instance)
        instance.start()
        vprint("Started, PID:", instance.pid)
    vprint("All blocks are started.")
  except (Exception, KeyboardInterrupt) as e:
    print("Exception in start :", e)
    for instance in MasterBlock.instances:
      try:
        instance.stop()
      except Exception as e:
        print(e)
  if not start_loop:
    return
  vprint("Waiting for all blocks to be ready...")
  while not all(map(lambda x: x.status == "ready", MasterBlock.instances)):
    time.sleep(.1)
  vprint("All blocks ready, let's go !")
  launch(verbose=verbose)

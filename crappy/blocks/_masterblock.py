# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MasterBlock MasterBlock
# @{

## @file _masterblock.py
# @brief Main class for block architecture. All blocks should inherit this class.
#
# @authors Victor Couty
# @version 1.1
# @date 07/01/2017
from __future__ import print_function

from multiprocessing import Process, Pipe
from ..links._link import TimeoutError

import time


def uncomp(data):
  """Used to uncompact data: only keeps the last data of the list (if any)"""
  if data is None:
    return
  for k in data:
    try:
      data[k] = data[k][-1]
    except (TypeError, IndexError):  # Already uncompacted or list empty
      pass
  return data


class MasterBlock(Process):
  """
  This represent a Crappy block, it must be parent of all the blocks.
  Methods:
    main()
      It must not take any arg, it is where you define the main loop of the block
      If not overriden, will raise an error

    add_[in/out]put(Link object)
      Add a link as [in/out]put

    prepare()
      This method will be called inside the new process but before actually starting the main loop of the program
      Use it for all the tasks to be done before starting the main loop (can be empty)
    start()
      This is the same start method as Process.start: it starts the process, so the initialization (defined in prepare method) will be done, but NOT the main loop

    launch(t0)
      Once the process is started, calling launch will set the starting time and actually start the main method.
      If the block was not started yet, it will be done automatically.
      t0: time to set as starting time of the block (mandatory) (in seconds after epoch)

    status
      Property that can be accessed both in the process or from the parent
        "idle": Block not started yet
        "initializing": start was called and prepare is not over yet
        "ready": prepare is over, waiting to start main by calling launch
        "running": main is running
        "done": main is over
        start and launch method will return instantly

  """
  instances = []

  def __init__(self):
    Process.__init__(self)
    # MasterBlock.instances.append(self)
    self.outputs = []
    self.inputs = []
    # This pipe allows to send 2 essential signals:
    # pipe1->pipe2 is to start the main function and set t0
    # pipe2->pipe1 to set process status to the parent
    self.pipe1, self.pipe2 = Pipe()
    self._status = "idle"
    self.in_process = False  # To know if we are in the process or not

  def __new__(cls, *args, **kwargs):
    instance = super(MasterBlock, cls).__new__(cls, *args, **kwargs)
    MasterBlock.instances.append(instance)
    return instance

  def __del__(self):
    Masterblock.instances.remove(self)

  def run(self):
    try:
      self.in_process = True  # we are in the process
      self.status = "initializing"
      self.prepare()
      self.status = "ready"
      self.t0 = self.pipe2.recv()  # Wait for parent to tell me to start the main
      self.status = "running"
      self.main()
      self.status = "done"
    except Exception as e:
      print("[%r] Exception caught:" % self, e)
      raise
    except KeyboardInterrupt:
      print("[%r] Keyboard interrupt received" % self)
      raise

  def start(self):
    """
    This will NOT execute the main, only start the process
    prepare will be called but not main !
    """
    self._status = "initializing"
    Process.start(self)

  def launch(self, t0):
    """
    To start the main method, will call start if needed
    """
    if self.status == "idle":
      print(self, ": Called launch on unprepared process!")
      self.start()
    self.pipe1.send(t0)  # asking to start main in the process

  @property
  def status(self):
    """
    Returns the status of the block, from the process itself or the parent
    """
    if not self.in_process:
      while self.pipe1.poll():
        self._status = self.pipe1.recv()
    return self._status

  @status.setter
  def status(self, s):
    assert self.in_process, "Cannot set status from outside of the process!"
    self.pipe2.send(s)
    self._status = s

  def main(self):
    """The method that will be run when .launch() is called"""
    raise NotImplementedError("Override me!")

  def prepare(self):
    """The first code to be run in the new process, will only be called
    once and before the actual start of the main launch of the blocks
    can do nothing"""
    pass

  def send(self, data):
    for o in self.outputs:
      o.send(data)

  def recv(self, in_id=0, blocking=True, uncompact=False):
    if uncompact:
      return uncomp(self.inputs[in_id].recv(blocking))
    return self.inputs[in_id].recv(blocking)

  def recv_any(self, blocking=True, uncompact=False):
    """Tries to recv data from the first waiting input, can be blocking
    or non blocking (will then return None if no data is waiting)"""
    while True:
      for i in self.inputs:
        if i.poll():
          if uncompact:
            return uncomp(i.recv())
          return i.recv()
      if not blocking:
        break

  def recv_last(self, uncompact=False):
    """Will get the latest data in each pipe, dropping all the other and
    then combines them. Necessarily non blocking"""
    data = None
    for l in self.inputs:
      if data:
        new = l.recv_last()
        if new:
          data.update(new)
      else:
        data = l.recv_last()
    if uncompact:
      return uncomp(data)
    return data

  def clear_inputs(self):
    """Will clear all the inputs of the block"""
    for l in self.inputs:
      l.clear()

  def add_output(self, o):
    self.outputs.append(o)

  def add_input(self, i):
    self.inputs.append(i)

  def stop(self):
    try:
      self.terminate()
    except Exception as e:
      print(self, "Could not terminate:", e)

  def __repr__(self):
    return str(type(self)) + " (" + self.status + ")"


def delay(s):
  time.sleep(s)

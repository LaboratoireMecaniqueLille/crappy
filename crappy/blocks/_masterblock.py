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
from collections import OrderedDict
from time import sleep, time, localtime, strftime


class CrappyStop(Exception):
  pass


class MasterBlock(Process):
  """
  This represent a Crappy block, it must be parent of all the blocks.
  Methods:
    main()
      It must not take any arg, it is where you define the main loop of
      the block. If not overriden, will raise an error

    add_[in/out]put(Link object)
      Add a link as [in/out]put

    prepare()
      This method will be called inside the new process but before actually
      starting the main loop of the program.  Use it for all the tasks to be
      done before starting the main loop (can be empty).
    start()
      This is the same start method as Process.start: it starts the process,
      so the initialization (defined in prepare method)
      will be done, but NOT the main loop.

    launch(t0)
      Once the process is started, calling launch will set the starting time
      and actually start the main method.  If the block was not started yet,
      it will be done automatically.
      t0: time to set as starting time of the block
      (mandatory, in seconds after epoch)

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
    self.in_process = True  # we are in the process
    self.status = "initializing"
    self.prepare()
    self.status = "ready"
    # Wait for parent to tell me to start the main
    self.t0 = self.pipe2.recv()
    self.status = "running"
    self.begin()
    try:
      self.main()
      self.status = "done"
    except Exception as e:
      print("[%r] Exception caught:" % self, e)
      self.finish()
      self.status = "error"
      raise
    except KeyboardInterrupt:
      print("[%r] Keyboard interrupt received" % self)
    self.finish()
    self.status = "done"

  @classmethod
  def get_status(cls):
    return map(lambda x: x.status, cls.instances)

  @classmethod
  def all_are(cls, s):
    """
    Returns true only if all processes status are s
    """
    return len(set(cls.get_status())) == 1 and s in cls.get_status()

  @classmethod
  def prepare_all(cls, verbose=True):
    """
    Starts all the blocks processes (block.prepare), but not the main loop
    """
    if verbose:
      def vprint(*args):
        print("[prepare]", *args)
    else:
      vprint = lambda *x: None
    vprint("Starting the blocks...")
    for instance in cls.instances:
      vprint("Starting", instance)
      instance.start()
      vprint("Started, PID:", instance.pid)
    vprint("All processes are started.")

  @classmethod
  def launch_all(cls, t0=None, verbose=True, wait=True):
    if verbose:
      def vprint(*args):
        print("[launch]", *args)
    else:
      vprint = lambda *x: None
    if not cls.all_are('ready'):
      vprint("Waiting for all blocks to be ready...")
    while not cls.all_are('ready'):
      sleep(.1)
    vprint("All blocks ready, let's go !")
    if not t0:
      t0 = time()
    vprint("Setting t0 to", strftime("%d %b %Y, %H:%M:%S", localtime(t0)))
    for instance in cls.instances:
      instance.launch(t0)
    t1 = time()
    vprint("All blocks loop started. It took", (t1 - t0) * 1000, "ms")
    if not wait:
      return
    try:
      # Keep running
      while True:
        sleep(31536000)  # 1 year, just to be sure
    except KeyboardInterrupt:
      print("Main proccess got keyboard interrupt!")
      if not cls.all_are('running'):
        print('Waiting for all processes to finish')
      while 'running' in cls.get_status():
        sleep(.1)
      print("Crappy terminated gracefully")

  @classmethod
  def start_all(cls, t0=None, verbose=True, wait=True):
    cls.prepare_all(verbose)
    cls.launch_all(t0, verbose, wait)

  @classmethod
  def stop_all(cls, verbose=True):
    """
    Stops all the blocks (crappy.stop)
    """
    if verbose:
      def vprint(*args):
        print("[stop]", *args)
    else:
      vprint = lambda *x: None
    vprint("Stopping the blocks...")
    for instance in cls.instances:
      if instance.status == 'running':
        vprint("Stopping", instance, "(PID:{})".format(instance.pid))
        instance.stop()
    vprint("All blocks are stopped.")

  def begin(self):
    """
    If main is not overriden, this method will be called first, before
    entering the main loop
    """
    pass

  def finish(self):
    """
    If main is not overriden, this method will be called upon exit or after
    a crash.
    """
    pass

  def loop(self):
    raise NotImplementedError('You must override loop or main in' + str(self))

  def main(self):
    while not self.pipe2.poll():
      self.loop()
    print("[%r] Got stop signal, interrupting..." % self)

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
      self.pipe2.send(
        self._status)  # If another process tries to get the status
    return self._status

  @status.setter
  def status(self, s):
    assert self.in_process, "Cannot set status from outside of the process!"
    self.pipe2.send(s)
    self._status = s

  def prepare(self):
    """The first code to be run in the new process, will only be called
    once and before the actual start of the main launch of the blocks
    can do nothing"""
    pass

  def send(self, data):
    """
    Send has 2 ways to operate: you can either build the ordered dict yourself
    or you can define self.labels (usually time first) and call send with a
    list. It will then map them to the dict.
    Note that ONLY OrderedDict can go through links
    """
    if type(data) == OrderedDict:
      pass
    elif type(data) == list:
      data = OrderedDict(zip(self.labels, data))
    elif data == 'stop':
      pass
    else:
      raise IOError("Trying to send a " + str(type(data)) + " in a link!")
    for o in self.outputs:
      o.send(data)

  def get_last(self, num=None):
    """
    Unlike the recv methods of Link, get_last is NOT guaranteed to return
    all the data going through the links! It is meant to get the latest values,
    discarding all the previous one (for a displayer for example)
    Its mode of operation is completely different since it can operate on
    multiple inputs at once. num is a list containing all the concerned inputs.
    The first call may be blocking until it receives data, all the others will
    return instantaneously, giving the latest known reading
    If num is None, it will operate on all the input link at once
    """
    if not hasattr(self, '_last_values'):
      self._last_values = []
      for i in self.inputs:
        self._last_values.append(None)
    if num is None:
      num = range(len(self.inputs))
    for i in num:
      if self._last_values[i] is None:
        self._last_values[i] = self.inputs[i].recv()
      while self.inputs[i].poll():
        self._last_values[i] = self.inputs[i].recv()
    ret = OrderedDict()
    for i in num:
      ret.update(self._last_values[i])
    return ret

  def drop(self, num=None):
    """Will clear the inputs of the blocks, performs like get_last
    but returns None instantly"""
    if num is None:
      num = range(len(self.inputs))
    for n in num:
      self.inputs[n].clear()

  def add_output(self, o):
    self.outputs.append(o)

  def add_input(self, i):
    self.inputs.append(i)

  def stop(self):
    if self.status != 'running':
      return
    print('[%r] Stopping' % self)
    self.pipe1.send(0)
    t = time()
    while self.status != 'done' and time() - t < 2:
      sleep(.1)
      try:
        self.inputs[0].send('stop')
        print("DEBUG: sent stop!")
      except IndexError:
        pass
    if self.status != "done":
      print('[%r] Could not stop properly, terminating' % self)
      self.terminate()
    else:
      print("[%r] Stopped correctly" % self)

  def __repr__(self):
    return str(type(self)) + " (" + self.status + ")"

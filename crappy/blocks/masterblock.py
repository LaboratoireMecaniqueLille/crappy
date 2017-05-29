# coding: utf-8
from __future__ import print_function,division

from sys import platform
from multiprocessing import Process, Pipe
from collections import OrderedDict
from time import sleep, time, localtime, strftime

from .._global import CrappyStop


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
        "error": An error occured and the block stopped
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
    MasterBlock.instances.remove(self)

  def run(self):
    self.in_process = True  # we are in the process
    self.status = "initializing"
    self.prepare()
    self.status = "ready"
    # Wait for parent to tell me to start the main
    self.t0 = self.pipe2.recv()
    self.status = "running"
    try:
      self.begin()
      self._MB_last_t = time()
      self._MB_last_FPS = self._MB_last_t
      self._MB_loops = 0
      self.main()
      self.status = "done"
    except CrappyStop:
      print("[%r] Encountered CrappyStop Exception, terminating" % self)
    except KeyboardInterrupt:
      print("[%r] Keyboard interrupt received" % self)
    except Exception as e:
      print("[%r] Exception caught:" % self, e)
      self.finish()
      self.status = "error"
      raise
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
    l = cls.get_status()
    return len(set(l)) == 1 and s in l

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
  def launch_all(cls, t0=None, verbose=True):
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
    try:
      # Keep running
      l = cls.get_status()
      while not "done" in l or "error" in l:
        l = cls.get_status()
        sleep(1)
    except KeyboardInterrupt:
      print("Main proccess got keyboard interrupt!")
      # It will automatically propagate to the blocks processes
    if not cls.all_are('running'):
      print('Waiting for all processes to finish')
    t = time()
    while 'running' in cls.get_status() and time() - t < 3:
      sleep(.1)
    if cls.all_are('done'):
      print("Crappy terminated gracefully")
    else:
      print("Crappy terminated, blocks status:")
      for b in cls.instances:
        print(b)

  @classmethod
  def start_all(cls, t0=None, verbose=True):
    cls.prepare_all(verbose)
    cls.launch_all(t0, verbose)

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
      self.handle_freq()
    print("[%r] Got stop signal, interrupting..." % self)

  def handle_freq(self):
    """
    For block with a given number of loops/s (use freq attr to set it)
    """
    self._MB_loops += 1
    t = time()
    if hasattr(self,'freq') and self.freq > 0:
      d = t-self._MB_last_t+1/self.freq
      while d > 0:
        t = time()
        d = self._MB_last_t+1/self.freq-t
        sleep(max(0,d/2-2e-3))# Ugly, yet simple and pretty efficient
    self._MB_last_t = t
    if hasattr(self,'verbose') and self.verbose and\
            self._MB_last_t - self._MB_last_FPS > 2:
        print("[%r] loops/s:"%self,
            self._MB_loops/(self._MB_last_t - self._MB_last_FPS))
        self._MB_loops = 0
        self._MB_last_FPS = self._MB_last_t

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
      #If another process tries to get the status
      if 'linux' in platform:
        self.pipe2.send(self._status)

      #Somehow the previous line makes crappy hang on Windows, no idea why
      #It is not critical, but it means that process status can now only
      #be read from the process itself and ONE other process.
      #Luckily, only the parent (the main process) needs the status for now
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
    if isinstance(data, dict):
      pass
    elif isinstance(data, list):
      data = dict(zip(self.labels, data))
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
      self._last_values = [None]*len(self.inputs)
    if num is None:
      num = range(len(self.inputs))
    elif not isinstance(num,list):
      num = [num]
    for i in num:
      if self._last_values[i] is None:
        self._last_values[i] = self.inputs[i].recv()
      while self.inputs[i].poll():
        self._last_values[i] = self.inputs[i].recv()
    ret = {}
    for i in num:
      ret.update(self._last_values[i])
    return ret

  def get_all_last(self, num=None):
    """
    Almost the same as get_last, but will return all the data that goes
    through the links (in lists). Note that for the sake of returning at
    least one value per label, it MAY return a value more than once on
    successive calls. Also, if multiple links have the same label,
    only the last link's value will be kept
    """
    if not hasattr(self, '_all_last_values'):
      self._all_last_values = [None]*len(self.inputs)
    if num is None:
      num = range(len(self.inputs))
    elif not isinstance(num,list):
      num = [num]
    for i in num:
      if self._all_last_values[i] is None or self.inputs[i].poll():
        self._all_last_values[i] = self.inputs[i].recv_chunk()
      else:
        # Dropping all data (already sent on last call) except the last
        # to make sure the block has at least one value
        for key in self._all_last_values[i]:
          self._all_last_values[i][key] = [self._all_last_values[i][key][-1]]
    ret = {}
    for i in num:
      ret.update(self._all_last_values[i])
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
      for i in self.inputs:
        i.send('stop')
    if self.status != "done":
      print('[%r] Could not stop properly, terminating' % self)
      try:
        self.terminate()
      except:
        pass
    else:
      print("[%r] Stopped correctly" % self)

  def __repr__(self):
    return str(type(self)) + " (" + self.status + ")"

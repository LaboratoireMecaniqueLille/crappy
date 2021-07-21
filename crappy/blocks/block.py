# coding: utf-8

from sys import platform
from multiprocessing import Process, Pipe
from time import sleep, time, localtime, strftime
from weakref import WeakSet
from pickle import UnpicklingError

from .._global import CrappyStop

import subprocess


def renice(pid, niceness):
  """Function to renice a process.

  Warning:
    Only works in Linux.

  Note:
    The user must be allowed to use ``sudo`` to renice with a negative value.
    May thus ask for a password for negative values.
  """

  if niceness < 0:
    subprocess.call(['sudo', 'renice', str(niceness), '-p', str(pid)])
  else:
    subprocess.call(['renice', str(niceness), '-p', str(pid)])


class Block(Process):
  """This represent a Crappy block, it must be parent of all the blocks."""

  instances = WeakSet()

  def __init__(self):
    Process.__init__(self)
    # Block.instances.append(self)
    self.outputs = []
    self.inputs = []
    # This pipe allows to send 2 essential signals:
    # pipe1->pipe2 is to start the main function and set t0
    # pipe2->pipe1 to set process status to the parent
    self.pipe1, self.pipe2 = Pipe()
    self._status = "idle"
    self.in_process = False  # To know if we are in the process or not
    self.niceness = 0

  def __new__(cls, *args, **kwargs):
    instance = super().__new__(cls)
    Block.instances.add(instance)
    return instance

  @classmethod
  def reset(cls):
    cls.instances = WeakSet()

  def run(self):
    self.in_process = True  # we are in the process
    self.status = "initializing"
    try:
      self.prepare()
      self.status = "ready"
      # Wait for parent to tell me to start the main
      self.t0 = self.pipe2.recv()
      if self.t0 < 0:
        try:
          self.finish()
        except Exception:
          pass
        return
      self.status = "running"
      self.begin()
      self._MB_last_t = time()
      self._MB_last_FPS = self._MB_last_t
      self._MB_loops = 0
      self.main()
      self.status = "done"
    except CrappyStop:
      print("[%r] Encountered CrappyStop Exception, terminating" % self)
      self.status = "done"
      self.stop_all()
    except KeyboardInterrupt:
      print("[%r] Keyboard interrupt received" % self)
    except Exception as e:
      print("[%r] Exception caught:" % self, e)
      try:
        self.finish()
      except Exception:
        pass
      self.status = "error"
      sleep(1)  # To let downstream blocks process the data and avoid loss
      self.stop_all()
      raise
    self.finish()
    self.status = "done"

  @classmethod
  def get_status(cls):
    return [x.status for x in cls.instances]

  @classmethod
  def all_are(cls, s):
    """Returns :obj:`True` only if all processes status are `s`."""

    lst = cls.get_status()
    return len(set(lst)) == 1 and s in lst

  @classmethod
  def renice_all(cls, high_prio=True, **_):
    """Will renice all the blocks processes according to ``block.niceness``
    value.

    Note:
      If ``high_prio`` is :obj:`False`, blocks with a negative niceness value
      will be ignored.

      This is to avoid asking for the ``sudo`` password since only root can
      lower the niceness of processes.
    """

    if "win" in platform:
      # Not supported on Windows yet
      return
    for b in cls.instances:
      if b.niceness < 0 and high_prio or b.niceness > 0:
        print("[renice] Renicing", b.pid, "to", b.niceness)
        renice(b.pid, b.niceness)

  @classmethod
  def prepare_all(cls, verbose=True):
    """Starts all the blocks processes (``block.prepare``), but not the main
    loop."""

    if verbose:
      def vprint(*args):
        print("[prepare]", *args)
    else:
      def vprint(*_):
        return
    vprint("Starting the blocks...")
    for instance in cls.instances:
      vprint("Starting", instance)
      instance.start()
      vprint("Started, PID:", instance.pid)
    vprint("All processes are started.")

  @classmethod
  def launch_all(cls, t0=None, verbose=True, bg=False):
    if verbose:
      def vprint(*args):
        print("[launch]", *args)
    else:
      def vprint(*_):
        return
    if not cls.all_are('ready'):
      vprint("Waiting for all blocks to be ready...")
    while not cls.all_are('ready'):
      sleep(.1)
      if not all([i in ['ready', 'initializing', 'idle']
            for i in cls.get_status()]):
          print("Crappy failed to start!")
          for i in cls.instances:
            if i.status in ['ready', 'initializing']:
              i.launch(-1)
          cls.stop_all()
          return
          # raise RuntimeError("Crappy failed to start!")
    vprint("All blocks ready, let's go !")
    if not t0:
      t0 = time()
    vprint("Setting t0 to", strftime("%d %b %Y, %H:%M:%S", localtime(t0)))
    for instance in cls.instances:
      instance.launch(t0)
    t1 = time()
    vprint("All blocks loop started. It took", (t1 - t0) * 1000, "ms")
    if bg:
      return
    try:
      # Keep running
      lst = cls.get_status()
      while not ("done" in lst or "error" in lst):
        lst = cls.get_status()
        sleep(1)
    except KeyboardInterrupt:
      print("Main process got keyboard interrupt!")
      cls.stop_all()
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
        print(b, b.status)

  @classmethod
  def start_all(cls, t0=None, verbose=True, bg=False, high_prio=False):
    cls.prepare_all(verbose)
    if high_prio and any([b.niceness < 0 for b in cls.instances]):
      print("[start] High prio: root permission needed to renice")
    cls.renice_all(high_prio, verbose=verbose)
    cls.launch_all(t0, verbose, bg)

  @classmethod
  def stop_all(cls, verbose=True):
    """Stops all the blocks (``crappy.stop``)."""

    if verbose:
      def vprint(*args):
        print("[stop]", *args)
    else:
      def vprint(*_):
        return
    vprint("Stopping the blocks...")
    for instance in cls.instances:
      if instance.status == 'running':
        vprint("Stopping", instance)
        instance.stop()
    vprint("All blocks are stopped.")

  def begin(self):
    """If :meth:`main` is not overridden, this method will be called first,
    before entering the main loop."""

    pass

  def finish(self):
    """If :meth:`main` is not overridden, this method will be called upon exit
    or after a crash."""

    pass

  def loop(self):
    raise NotImplementedError('You must override loop or main in' + str(self))

  def main(self):
    """This is where you define the main loop of the block.

    Important:
      If not overridden, will raise an error.
    """

    while not self.pipe2.poll():
      self.loop()
      self.handle_freq()
    print("[%r] Got stop signal, interrupting..." % self)

  def handle_freq(self):
    """For block with a given number of `loops/s` (use ``freq`` attribute to
    set it)."""

    self._MB_loops += 1
    t = time()
    if hasattr(self, 'freq') and self.freq:
      d = self._MB_last_t + 1 / self.freq - t
      while d > 0:
        t = time()
        d = self._MB_last_t + 1 / self.freq - t
        sleep(max(0, d / 2 - 2e-3))  # Ugly, yet simple and pretty efficient
    self._MB_last_t = t
    if hasattr(self, 'verbose') and self.verbose and \
                self._MB_last_t - self._MB_last_FPS > 2:
      print("[%r] loops/s:" % self,
            self._MB_loops / (self._MB_last_t - self._MB_last_FPS))
      self._MB_loops = 0
      self._MB_last_FPS = self._MB_last_t

  def launch(self, t0):
    """To start the :meth:`main` method, will call :meth:`Process.start` if
    needed.

    Once the process is started, calling launch will set the starting time and
    actually start the main method.

    Args:
      t0 (:obj:`float`): Time to set as starting time of the block (mandatory,
        in seconds after epoch).
    """

    if self.status == "idle":
      print(self, ": Called launch on unprepared process!")
      self.start()
    self.pipe1.send(t0)  # asking to start main in the process

  @property
  def status(self):
    """Returns the status of the block, from the process itself or the parent.

    It can be:

      - `"idle"`: Block not started yet.
      - `"initializing"`: :meth:`start` was called and :meth:`prepare` is not
        over yet.
      - `"ready"`: :meth:`prepare` is over, waiting to start :meth:`main` by
        calling :meth:`launch`.
      - `"running"`: :meth:`main` is running.
      - `"done"`: :meth:`main` is over.
      - `"error"`: An error occurred and the block stopped.
    """

    if not self.in_process:
      while self.pipe1.poll():
        try:
          self._status = self.pipe1.recv()
        except (EOFError, UnpicklingError):
          if self._status == 'running':
            self._status = 'done'
      # If another process tries to get the status
      if 'linux' in platform:
        self.pipe2.send(self._status)

        # Somehow the previous line makes crappy hang on Windows, no idea why
        # It is not critical, but it means that process status can now only
        # be read from the process itself and ONE other process.
        # Luckily, only the parent (the main process) needs the status for now
    return self._status

  @status.setter
  def status(self, s):
    assert self.in_process, "Cannot set status from outside of the process!"
    self.pipe2.send(s)
    self._status = s

  def prepare(self):
    """This will be run when creating the process, but before the actual start.

    The first code to be run in the new process, will only be called once and
    before the actual start of the main launch of the blocks.

    It can remain empty and do nothing.
    """

    pass

  def send(self, data):
    """To send the data to all blocks downstream.

    Send has 2 ways to operate. You can either build the :obj:`dict` yourself,
    or you can define ``self.labels`` (usually time first) and call send with a
    :obj:`list`. It will then map them to the :obj:`dict`.

    Note:
      ONLY :obj:`dict` can go through links.
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

  def recv_all(self):
    """Receives new data from all the inputs (not as chunks).

    It will simply call :meth:`Pipe.recv` on all non empty links and return a
    single :obj:`dict`.

    Important:
      If the same label comes from multiple links, it may be overridden !
    """

    r = {}
    for i in self.inputs:
      if i.poll():
        r.update(i.recv())
    return r

  def poll(self):
    """Tells if any input link has pending data

    Returns True if l.poll() is True for any input link l.
    """

    return any((l.poll for l in self.inputs))

  def recv_all_last(self):
    """Like recv_all, but drops older data to return only the latest value

    This method avoids Pipe congestion that can be induced by recv_all when the
    receiving block is slower than the block upstream
    """

    r = {}
    for i in self.inputs:
      while i.poll():
        r.update(i.recv())
    return r

  def get_last(self, num=None):
    """To get the latest value of each labels from all inputs.

    Warning:
      Unlike the ``recv`` methods of :ref:`Link`, this method is NOT guaranteed
      to return all the data going through the links!

      It is meant to get the latest values, discarding all the previous ones
      (for a displayer for example).

      Its mode of operation is completely different since it can operate on
      multiple inputs at once.

    Args:
      num (:obj:`list`, optional): A :obj:`list` containing ll the concerned
        inputs. If :obj:`None` it will operate on all the input links at once.

    Note:
      The first call may be blocking until it receives data, all the others
      will return instantaneously, giving the latest known reading.
    """

    if not hasattr(self, '_last_values'):
      self._last_values = [None] * len(self.inputs)
    if num is None:
      num = range(len(self.inputs))
    elif not isinstance(num, list):
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
    """To get the data from all links of the block.

    It is almost the same as :meth:`get_last`, but will return all the data
    that goes through the links (as :obj:`list`).

    Also, if multiple links have the same label, only the last link's value
    will be kept.
    """

    if not hasattr(self, '_all_last_values'):
      self._all_last_values = [None] * len(self.inputs)
    if num is None:
      num = range(len(self.inputs))
    elif not isinstance(num, list):
      num = [num]
    for i in num:
      if self._all_last_values[i] is None or self.inputs[i].poll():
        self._all_last_values[i] = self.inputs[i].recv_chunk()
      else:
        # Dropping all data (already sent on last call) except the last
        # to make sure the block has at least one value
        for key in self._all_last_values[i]:
          self._all_last_values[i][key][:-1] = []
    ret = {}
    for i in num:
      ret.update(self._all_last_values[i])
    return ret

  def recv_all_delay(self, delay=None, poll_delay=.1):
    """Method to wait for data, but continuously reading all the links to make
    sure that it does not block.

    Return:
      A :obj:`list` where each entry is what would have been returned by
      :meth:`Link.recv_chunk` on each link.
    """

    if delay is None:
      delay = 1 / self.freq
    t = time()
    r = [{} for _ in self.inputs]
    last = t
    while True:
      sleep(max(0., poll_delay - time() + last))
      for lst, d in zip(self.inputs, r):
        if not lst.poll():
          continue
        new = lst.recv_chunk()
        for k, v in new.items():
          if k in d:
            d[k].extend(v)
          else:
            d[k] = v
      if time() - t > delay:
        break
    return r

  def drop(self, num=None):
    """Will clear the inputs of the blocks.

    This method performs like :meth:`get_last`, but returns :obj:`None`
    instantly.
    """

    if num is None:
      num = range(len(self.inputs))
    elif not isinstance(num, list):
      num = [num]
    for n in num:
      self.inputs[n].clear()

  def add_output(self, o):
    """Adds a :ref:`Link` as an output."""

    self.outputs.append(o)

  def add_input(self, i):
    """Adds a :ref:`Link` as an input."""

    self.inputs.append(i)

  def stop(self):
    if self.status != 'running':
      return
    print('[%r] Stopping' % self)
    self.pipe1.send(0)
    for i in self.inputs:
      i.send('stop')
    for i in range(10):
      if self.status == "done":
        break
      sleep(.05)
    # if self.status != "done":
    if self.status not in ['done', 'idle', 'error']:
      print('[%r] Could not stop properly, terminating' % self)
      try:
        self.terminate()
      except Exception:
        pass
    else:
      print("[%r] Stopped correctly" % self)

  def __repr__(self):
    return str(type(self)) + " (" + str(self.pid or "Not running") + ")"

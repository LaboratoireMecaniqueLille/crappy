# coding: utf-8

from platform import system
from multiprocessing import Process, Value, Barrier, Event
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import synchronize
from threading import BrokenBarrierError
from time import sleep, time
from weakref import WeakSet
from typing import Union, Optional, List, Dict, Any
from collections import defaultdict
import subprocess

from ..links import Link
from .._global import CrappyStop

# Todo:
#  Add a clean way to stop the blocks, using the keyboard or a button


class Block(Process):
  """This class constitutes the base object in Crappy.

  It is extremely versatile, an can perform a wide variety of actions during a
  test. Many Blocks are already defined in Crappy, but it is possible to define
  custom ones for specific purposes.

  It is a subclass of :obj:`multiprocessing.Process`, and is thus an
  independent process in Python. It communicates with other Blocks via
  :mod:`multiprocessing` objects.
  """

  instances = WeakSet()

  # The synchronization objects will be set later
  shared_t0: Optional[Synchronized] = None
  ready_barrier: Optional[synchronize.Barrier] = None
  start_event: Optional[synchronize.Event] = None
  stop_event: Optional[synchronize.Event] = None

  def __init__(self) -> None:
    """Sets the attributes and initializes the parent class."""

    super().__init__()

    # The lists of input and output links
    self.outputs: List[Link] = list()
    self.inputs: List[Link] = list()

    # Various objects that should be set by child classes
    self.niceness: int = 0
    self.labels: Optional[List[str]] = None
    self.freq = None
    self.verbose = False

    # The synchronization objects will be set later
    self._instance_t0: Optional[Synchronized] = None
    self._ready_barrier: Optional[synchronize.Barrier] = None
    self._start_event: Optional[synchronize.Event] = None
    self._stop_event: Optional[synchronize.Event] = None

    # Objects for displaying performance information about the block
    self._last_t: Optional[float] = None
    self._last_fps: Optional[float] = None
    self._n_loops: int = 0

    self._last_values = None

  def __new__(cls, *args, **kwargs):
    """Called when instantiating a new instance of a Block. Adds itself to
    the WeakSet of all blocks."""

    instance = super().__new__(cls)
    print('new')
    cls.instances.add(instance)
    return instance

  @classmethod
  def start_all(cls, allow_root: bool = False) -> None:
    """Method for starting a script with Crappy.

    It sets the synchronization objects for all the blocks, renices the
    corresponding processes and starts the blocks.

    The call to this method is blocking until Crappy finishes.

    Note:
      It is possible to have a finer grained control of the start of a Crappy
      script with the methods :meth:`prepare_all`, :meth:`renice_all` and
      :meth:`launch_all`.

    Args:
      allow_root: If set tu :obj:`True`, tries to renice the processes niceness
        with sudo privilege in Linux. It requires the Python script to be run
        with sudo privilege, otherwise it has no effect.
    """

    cls.prepare_all()
    cls.renice_all(allow_root)
    cls.launch_all()

  @classmethod
  def prepare_all(cls) -> None:
    """Creates the synchronization objects, shares them with the blocks, and
    starts the processes associated to the blocks.

    Once started with this method, the blocks will call their :meth:`prepare`
    method and then be blocked by a :obj:`multiprocessing.Barrier`.
    """

    # Setting all the synchronization objects at the class level
    cls.ready_barrier = Barrier(len(cls.instances) + 1)
    cls.start_event = Event()
    cls.stop_event = Event()
    cls.shared_t0 = Value('d', -1.0)

    # Passing the synchronization objects to each block
    for instance in cls.instances:
      instance._ready_barrier = cls.ready_barrier
      instance._instance_t0 = cls.shared_t0
      instance._stop_event = cls.stop_event
      instance._start_event = cls.start_event

    # Starting all the blocks
    for instance in cls.instances:
      instance.start()

  @classmethod
  def renice_all(cls, allow_root: bool) -> None:
    """On Linux and MacOS, renices the processes associated with the blocks.

    On Windows, does nothing.

    Args:
      allow_root: If set tu :obj:`True`, tries to renice the processes niceness
        with sudo privilege in Linux. It requires the Python script to be run
        with sudo privilege, otherwise it has no effect.
    """

    # There's no niceness on Windows
    if system() == "Windows":
      return

    # Renicing all the blocks
    for inst in cls.instances:
      # If root is not allowed then the minimum niceness is 0
      niceness = max(inst.niceness, 0 if not allow_root else -20)

      # System call for setting the niceness
      if niceness < 0:
        subprocess.call(['sudo', 'renice', str(niceness), '-p', str(inst.pid)])
      else:
        subprocess.call(['renice', str(niceness), '-p', str(inst.pid)])

  @classmethod
  def launch_all(cls) -> None:
    """The final method being called by the main process running a Crappy
    script.

    It unlocks all the Blocks by releasing the synchronization barrier, sets
    the shared t0 value, and then waits for all the Blocks to finish.

    In case an exception is raised, sets the stop event for warning the Blocks,
    waits for the Blocks to finish, and if they don't, terminates them.
    """

    try:
      # The barrier waits for the main process to be ready so that the
      # prepare_all and launch_all methods can be used separately for a finer
      # grained control
      print('waiting')
      cls.ready_barrier.wait()
      print('waited')

      # Setting t0 and telling all the block to start
      cls.shared_t0.value = time()
      cls.start_event.set()

      # The main process mustn't finish before all the blocks are done running
      print('joining')
      for inst in cls.instances:
        inst.join()

    except (BrokenBarrierError, KeyboardInterrupt):
      # Warning all the blocks if an exception was caught
      print('exception')
      cls.stop_event.set()
      t = time()

      # Waiting at most 3 seconds for all the blocks to finish
      while not all(not inst.is_alive() for inst in cls.instances):
        sleep(0.1)

        # After 3 seconds, killing the blocks that didn't stop
        if time() - t > 3:
          for inst in cls.instances:
            if inst.is_alive():
              inst.terminate()

  @classmethod
  def stop_all(cls) -> None:
    """Method for stopping all the Blocks by setting the stop event."""

    cls.stop_event.set()

  @classmethod
  def reset(cls) -> None:
    """Resets Crappy by emptying the WeakSet containing references to all the
    Blocks. Only useful for restarting Crappy from a script where Crappy was
    already started."""

    cls.instances = WeakSet()

  def run(self) -> None:
    """The method run by the Blocks when their process is started.

    It first calls :meth:`prepare`, then waits at the
    :obj:`multiprocessing.Barrier` for all Blocks to be ready, then calls
    :meth:`begin`, then :meth:`main`, and finally :meth:`finish`.
    
    If an exception is raised, sets the shared stop event to warn all the other
    Blocks.
    """

    try:

      # Running the preliminary actions before the test starts
      try:
        self.prepare()
      except (Exception,):
        # If exception is raised, breaking the barrier to warn the other blocks
        self._ready_barrier.abort()
        raise

      # Waiting for all blocks to be ready, except if the barrier was broken
      try:
        print('proc waiting')
        self._ready_barrier.wait()
        print('proc waited')
      except BrokenBarrierError:
        print('proc broken barrier')
        raise IOError

      # Waiting for t0 to be set, should take a few milliseconds at most
      self._start_event.wait(timeout=1)
      if not self._start_event.is_set():
        raise TimeoutError

      # Running the first loop
      self.begin()

      # Setting the attributes for counting the performance
      self._last_t = time()
      self._last_fps = self._last_t
      self._n_loops = 0

      # Running the main loop until told to stop
      self.main()

    # In case of an exception, telling the other blocks to stop
    except (IOError, Exception) as e:
      print('proc stop event', e)
      self._stop_event.set()
      raise
    except (KeyboardInterrupt, CrappyStop) as e:
      print('proc stop event', e)
      self._stop_event.set()

    # In all cases, trying to properly close the block
    finally:
      print('proc finished')
      self.finish()

  def main(self) -> None:
    """The main loop of the :meth:`run` method. Repeatedly calls the
    :meth:`loop` method and manages the looping frequency."""

    # Looping until told to stop or an error occurs
    while not self._stop_event.is_set():
      self.loop()
      self._handle_freq()
    print('proc exiting main')

  def prepare(self) -> None:
    """This method should perform any action required for initializing the
    Block before the test starts.

    For example, it can open a network connection, create a file, etc. It is
    also fine for this method not to be overriden if there's no particular
    action to perform.

    Note that this method is called once the process associated to the Block
    has been started.
    """

    ...

  def begin(self) -> None:
    """This method can be considered as the first loop of the test, and is
    called before the :meth:`loop` method.

    It allows to perform initialization actions that cannot be achieved in the
    :meth:`prepare` method.
    """

    ...

  def loop(self) -> None:
    """This method is the core of the Block. It is called repeatedly during the
    test, until the test stops or an error occurs.

    Only in this method should data be sent to downstream blocks, or received
    from upstream blocks.

    Although it is possible not to override this method, that has no practical
    interest and this method should always be rewritten.
    """

    print(f"[Block {type(self).__name__}] Loop method not defined, this block "
          f"does nothing !")

  def finish(self) -> None:
    """This method should perform any action required for properly ending the
    test.

    For example, it can close a file or disconnect from a network. It is also
    fine for this method not to be overriden if no particular action needs to
    be performed.

    Note that this method should normally be called even in case an error
    occurs, although that cannot be guaranteed.
    """

    ...

  def _handle_freq(self) -> None:
    """This method ensures that the Block loops at the desired frequency, or as
    fast as possible if the requested frequency cannot be achieved.

    It also displays the looping frequency of the Block if requested by the
    user. If no looping frequency is specified, the Block will loop as fast as
    possible.
    """

    self._n_loops += 1

    # Only handling frequency if requested
    if self.freq is not None:
      t = time()

      # Correcting the error of the sleep function through a recursive approach
      # The last 2 milliseconds are in free loop
      remaining = self._last_t + 1 / self.freq - t
      while remaining > 0:
        t = time()
        remaining = self._last_t + 1 / self.freq - t
        sleep(max(0., remaining / 2 - 2e-3))

      self._last_t = t

    # Printing frequency every 2 seconds
    if self.verbose and self._last_t - self._last_fps > 2:
      print(f"{self} loops/s: "
            f"{self._n_loops / (self._last_t - self._last_fps)}")
      self._n_loops = 0
      self._last_fps = self._last_t

  @property
  def t0(self) -> float:
    """Returns the value of t0, the exact moment when the test started that is
    shared between all the Blocks."""

    if self._instance_t0 is not None and self._instance_t0.value > 0:
      return self._instance_t0.value
    else:
      raise ValueError("t0 not set yet !")

  def add_output(self, link: Link) -> None:
    """Adds an output link to the list of output links of the Block."""

    self.outputs.append(link)

  def add_input(self, link: Link) -> None:
    """Adds an input link to the list of input links of the Block."""

    self.inputs.append(link)

  def send(self, data: Union[Dict[str, Any], List[Any]]) -> None:
    """Ensures that the data to send is formatted as a :obj:`dict`, and sends
    it in all the downstream links."""

    # Building the dict to send from the data and labels if the data is a list
    if isinstance(data, list):
      if not self.labels:
        raise IOError("trying to send data as a list but no labels are "
                      "specified ! Please add a self.labels attribute.")
      data = dict(zip(self.labels, data))

    # Making sure the data is being sent as a dict
    elif not isinstance(data, dict):
      raise IOError(f"Trying to send a {type(data)} in a link!")

    # Sending the data to the downstream blocks
    for link in self.outputs:
      link.send(data)

  def data_available(self) -> bool:
    """Returns :obj:`True` if there's data available for reading in at least
    one of the input Links."""

    return self.inputs and any(link.poll() for link in self.inputs)

  def recv_data(self) -> Dict[str, Any]:
    """Reads the first available values from each incoming Link and returns
    them all in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Also, the returned values are the oldest available in the links. See
    :meth:`recv_last_data` for getting the newest available values.

    Important:
      If data is received over a same label from different links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a single value
      for each key (usually a :obj:`float` or a :obj:`str`).
    """

    ret = dict()

    for link in self.inputs:
      ret.update(link.recv())

    return ret

  def recv_last_data(self, fill_missing: bool = True) -> Dict[str, Any]:
    """Reads all the available values from each incoming Link, and returns
    the newest ones in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Important:
      If data is received over a same label from different links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.

    Args:
      fill_missing: If :obj:`True`, fills up the missing data for the known
        labels. This way, the last value received from all known labels is
        always returned. It can of course not fill up missing data for labels
        that haven(t been received yet.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a single value
      for each key (usually a :obj:`float` or a :obj:`str`).
    """

    # Initializing the buffer storing the last received values
    if self._last_values is None:
      self._last_values = [dict() for _ in self.inputs]

    ret = dict()

    # Storing the received values in the return dict and in the buffer
    for link, buffer in zip(self.inputs, self._last_values):
      data = link.recv_last()
      ret.update(data)
      buffer.update(data)

    # If requested, filling up the missing values in the return dict
    if fill_missing:
      for buffer in self._last_values:
        ret.update(buffer)

    return ret

  def recv_all_data(self,
                    delay: Optional[float] = None,
                    poll_delay: float = 0.1) -> Dict[str, List[Any]]:
    """Reads all the available values from each incoming Link, and returns
    them all in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Important:
      If data is received over a same label from different links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.
      See the :meth:`recv_all_data_raw` method for receiving data with no loss.

    Warning:
      As the time label is (normally) shared between all Blocks, the values
      returned for this label will be inconsistent and shouldn't be used !

    Args:
      delay: If given specifies a delay, as a :obj:`float`, during which the
        method acquired data before returning. All the data received during
        this delay is saved and returned. Otherwise, just reads all the
        available data and returns as soon as it is exhausted.
      poll_delay: If the ``delay`` argument is given, the Links will be polled
        once every this value seconds. It ensures that the method doesn't spam
        the CPU in vain.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a :obj:`list`
      of received values for each key. The first item in the list is the oldest
      one available in the link, the last item is the newest available.
    """

    ret = defaultdict(list)
    t0 = time()

    # If simple recv_all, just receiving from all input links
    if delay is None:
      for link in self.inputs:
        ret.update(link.recv_chunk())

    # Otherwise, receiving during the given period
    else:
      while time() - t0 < delay:
        last_t = time()
        # Updating the list of received values
        for link in self.inputs:
          data = link.recv_chunk()
          for label, values in data.items():
            ret[label].extend(values)
        # Sleeping to avoid useless CPU usage
        sleep(max(0., last_t + poll_delay - time()))

    # Returning a dict, not a defaultdict
    return dict(ret)

  def recv_all_data_raw(self,
                        delay: Optional[float] = None,
                        poll_delay: float = 0.1) -> List[Dict[str, List[Any]]]:
    """Reads all the available values from each incoming Link, and returns
    them separately in a list of dicts.

    Unlike :meth:`recv_all_data` this method does not fuse the received data
    into a single :obj:`dict`, so it is guaranteed to return all the available
    data with no loss.

    Args:
      delay: If given specifies a delay, as a :obj:`float`, during which the
        method acquired data before returning. All the data received during
        this delay is saved and returned. Otherwise, just reads all the
        available data and returns as soon as it is exhausted.
      poll_delay: If the ``delay`` argument is given, the Links will be polled
        once every this value seconds. It ensures that the method doesn't spam
        the CPU in vain.

    Returns:
      A :obj:`list` containing :obj:`dict`, whose keys are the received labels
      and with a :obj:`list` of received value for each key.
    """

    ret = [defaultdict(list) for _ in self.inputs]
    t0 = time()

    # If simple recv_all, just receiving from all input links
    if delay is None:
      for dic, link in zip(ret, self.inputs):
        dic.update(link.recv_chunk())

    # Otherwise, receiving during the given period
    else:
      while time() - t0 < delay:
        last_t = time()
        # Updating the list of received values
        for dic, link in zip(ret, self.inputs):
          data = link.recv_chunk()
          for label, values in data.items():
            dic[label].extend(values)
        # Sleeping to avoid useless CPU usage
        sleep(max(0., last_t + poll_delay - time()))

    return [dict(dic) for dic in ret]

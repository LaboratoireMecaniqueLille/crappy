# coding: utf-8

from platform import system
from multiprocessing import Process, Value, Barrier, Event, Queue, \
  get_start_method
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import synchronize, queues
from threading import BrokenBarrierError, Thread
from queue import Empty
import logging
import logging.handlers
from time import sleep, time
from weakref import WeakSet
from typing import Union, Optional, List, Dict, Any
from collections import defaultdict
import subprocess
from sys import stdout, stderr, argv
from pathlib import Path

from ..links import Link
from .._global import LinkDataError, StartTimeout, PrepareError, \
  T0NotSetError, GeneratorStop, ReaderStop, CameraPrepareError, \
  CameraRuntimeError

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
  logger: Optional[logging.Logger] = None
  log_queue: Optional[queues.Queue] = None
  log_thread: Optional[Thread] = None
  thread_stop = False

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

    # The objects for logging will be set later
    self._log_queue: Optional[queues.Queue] = None
    self._logger: Optional[logging.Logger] = None
    self._log_level = logging.INFO

    # Objects for displaying performance information about the block
    self._last_t: Optional[float] = None
    self._last_fps: Optional[float] = None
    self._n_loops: int = 0

    self._last_values = None

  def __new__(cls, *args, **kwargs):
    """Called when instantiating a new instance of a Block. Adds itself to
    the WeakSet of all blocks."""

    instance = super().__new__(cls)
    cls.instances.add(instance)
    return instance

  @classmethod
  def start_all(cls,
                allow_root: bool = False,
                log_level: int = 20) -> None:
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
      log_level: An :obj:`int` indicating the logging level to use when running
        the script. Default is `20` for level INFO, other levels are `10` for
        DEBUG, `30` for WARNING, `40` for ERROR and `50` for CRITICAL. The
        verbosity of the DEBUG level is really high, so it should only be used
        when needed.
    """

    cls.prepare_all(log_level)
    cls.renice_all(allow_root)
    cls.launch_all()

  @classmethod
  def prepare_all(cls, log_level: int = 20) -> None:
    """Creates the synchronization objects, shares them with the blocks, and
    starts the processes associated to the blocks.

    Also initializes the logger for the Crappy script.

    Once started with this method, the blocks will call their :meth:`prepare`
    method and then be blocked by a :obj:`multiprocessing.Barrier`.

    Args:
      log_level: An :obj:`int` indicating the logging level to use when running
        the script. Default is `20` for level INFO, other levels are `10` for
        DEBUG, `30` for WARNING, `40` for ERROR and `50` for CRITICAL. The
        verbosity of the DEBUG level is really high, so it should only be used
        when needed.
    """

    try:

      # Initializing the logger and displaying the first messages
      cls._set_logger(log_level)
      cls.logger.log(logging.INFO,
                     "===================== CRAPPY =====================")
      cls.logger.log(logging.INFO, f'Starting the script {argv[0]}\n')
      cls.logger.log(logging.INFO, 'Logger configured')

      # Setting all the synchronization objects at the class level
      cls.ready_barrier = Barrier(len(cls.instances) + 1)
      cls.shared_t0 = Value('d', -1.0)
      cls.start_event = Event()
      cls.stop_event = Event()
      cls.logger.log(logging.INFO, 'Multiprocessing synchronization objects '
                                   'set for main process')

      # Initializing the objects required for logging
      cls.log_thread = Thread(target=cls._log_target)
      cls.log_queue = Queue()
      cls.log_thread.start()
      cls.logger.log(logging.INFO, 'Logger thread started')

      # Passing the synchronization and logging objects to each block
      for instance in cls.instances:
        instance._ready_barrier = cls.ready_barrier
        instance._instance_t0 = cls.shared_t0
        instance._stop_event = cls.stop_event
        instance._start_event = cls.start_event
        instance._log_queue = cls.log_queue
        instance._log_level = log_level
        cls.logger.log(logging.INFO, f'Multiprocessing synchronization objects'
                                     f' set for {instance.name} Block')

      # Starting all the blocks
      for instance in cls.instances:
        instance.start()
        cls.logger.log(logging.INFO, f'Started the {instance.name} Block')

    except KeyboardInterrupt:
      cls.logger.log(logging.INFO, 'KeyboardInterrupt caught while running '
                                   'prepare_all')
      cls._exception()

  @classmethod
  def renice_all(cls, allow_root: bool) -> None:
    """On Linux and MacOS, renices the processes associated with the blocks.

    On Windows, does nothing.

    Args:
      allow_root: If set tu :obj:`True`, tries to renice the processes niceness
        with sudo privilege in Linux. It requires the Python script to be run
        with sudo privilege, otherwise it has no effect.
    """

    try:

      # There's no niceness on Windows
      if system() == "Windows":
        cls.logger.log(logging.INFO, 'Not renicing processes on Windows')
        return

      # Renicing all the blocks
      cls.logger.log(logging.INFO, 'Renicing processes')
      for inst in cls.instances:
        # If root is not allowed then the minimum niceness is 0
        niceness = max(inst.niceness, 0 if not allow_root else -20)

        # System call for setting the niceness
        if niceness < 0:
          subprocess.call(['sudo', 'renice', str(niceness), '-p',
                           str(inst.pid)], stdout=subprocess.DEVNULL)
          cls.logger.log(logging.INFO, f"Reniced process {inst.name} with PID "
                                       f"{inst.pid} to niceness {niceness} "
                                       f"with sudo privilege")
        else:
          subprocess.call(['renice', str(niceness), '-p', str(inst.pid)],
                          stdout=subprocess.DEVNULL)
          cls.logger.log(logging.INFO, f"Reniced process {inst.name} with PID "
                                       f"{inst.pid} to niceness {niceness}")

    except KeyboardInterrupt:
      cls.logger.log(logging.INFO, 'KeyboardInterrupt caught while running '
                                   'renice_all')
      cls._exception()

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
      cls.logger.log(logging.INFO, 'Waiting for all Blocks to be ready')
      cls.ready_barrier.wait()
      cls.logger.log(logging.INFO, 'All Blocks ready now')

      # Setting t0 and telling all the block to start
      cls.shared_t0.value = time()
      cls.logger.log(logging.INFO, f'Start time set to {cls.shared_t0.value}s')
      cls.start_event.set()
      cls.logger.log(logging.INFO, 'Start event set, all Blocks can now start')

      # The main process mustn't finish before all the blocks are done running
      cls.logger.log(logging.INFO, 'Main process done, waiting for all Blocks '
                                   'to finish')
      for inst in cls.instances:
        inst.join()
        cls.logger.log(logging.INFO, f'{inst.name} finished by itself')

      cls.logger.log(logging.INFO, 'All Blocks done, Crappy terminated '
                                   'gracefully\n')
      cls.thread_stop = True

    # A Block crashed while preparing
    except BrokenBarrierError:
      cls.logger.log(logging.ERROR, "Exception raised in a Block while "
                                    " waiting for all Blocks to be ready, "
                                    "stopping")
      cls._exception()
    # The user ended the script while preparing
    except KeyboardInterrupt:
      cls.logger.log(logging.INFO, 'KeyboardInterrupt caught while running '
                                   'launch_all')
      cls._exception()
    # Another exception occurred
    except (Exception,) as exc:
      cls.logger.exception("Caught exception while running launch_all, "
                           "aborting", exc_info=exc)
      cls._exception()

  @classmethod
  def _exception(cls) -> None:
    """This method is called when an exception is caught in the main process.

    It waits for all the Blocks to end, and kills them if they don't stop by
    themselves. Also stops the thread managing the logging.
    """

    cls.stop_event.set()
    cls.logger.log(logging.INFO, 'Stop event set, waiting for all Blocks to '
                                 'finish')
    t = time()

    # Waiting at most 3 seconds for all the blocks to finish
    while cls.instances and not all(not inst.is_alive() for inst
                                    in cls.instances):
      sleep(0.1)
      cls.logger.log(logging.DEBUG, "All Blocks not stopped yet")

      # After 3 seconds, killing the blocks that didn't stop
      if time() - t > 3:
        cls.logger.log(logging.WARNING, 'All Blocks not stopped, terminating '
                                        'the living ones')
        for inst in cls.instances:
          if inst.is_alive():
            inst.terminate()
            cls.logger.log(logging.WARNING, f'Block {inst.name} terminated')
          else:
            cls.logger.log(logging.INFO, f'Block {inst.name} done')

    cls.logger.log(logging.INFO, 'All Blocks done, Crappy terminated '
                                 'gracefully\n')
    cls.thread_stop = True

  @classmethod
  def _set_logger(cls, log_level: int = 20) -> None:
    """Initializes the logging for the main process.

    It creates two Stream Loggers, one for the info and debug levels displaying
    on stdout and one for the other levels displaying on stderr. It also
    creates a File Logger for saving the log to a log file.

    The levels WARNING and above are always being displayed in the terminal, no
    matter what the user chooses. Similarly, the INFO log level and above are
    always being saved to the log file.

    log_level: An :obj:`int` indicating the logging level to use when running
      the script. Default is `20` for level INFO, other levels are `10` for
      DEBUG, `30` for WARNING, `40` for ERROR and `50` for CRITICAL. The
      verbosity of the DEBUG level is really high, so it should only be used
      when needed.
    """

    log_level = 10 * int(round(log_level / 10, 0))

    # The logger handling all messages
    crappy_log = logging.getLogger('crappy')
    crappy_log.setLevel(logging.DEBUG)

    # The two handlers for displaying messages in the console
    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler_err = logging.StreamHandler(stream=stderr)

    # Getting the path to Crappy's temporary folder
    if system() in ('Linux', 'Darwin'):
      log_path = Path('/tmp/crappy')
    elif system() == 'Windows':
      log_path = Path.home() / 'AppData' / 'Local' / 'Temp' / 'crappy'
    else:
      log_path = None

    # Creating Crappy's temporary folder if needed
    if log_path is not None:
      try:
        log_path.mkdir(parents=False, exist_ok=True)
      except FileNotFoundError:
        log_path = None

    # This handler writes the log messages to a log file
    if log_path is not None:
      if log_level > 10:
        file_handler = logging.handlers.RotatingFileHandler(
          log_path / 'logs.txt', maxBytes=1000000, backupCount=5)
      else:
        file_handler = logging.FileHandler(log_path / 'logs_debug.txt',
                                           mode='w')
    else:
      file_handler = None

    # Setting the log levels for the handlers
    stream_handler.setLevel(log_level)
    stream_handler.addFilter(cls._stdout_filter)
    stream_handler_err.setLevel(max(logging.WARNING, log_level))
    if file_handler is not None:
      file_handler.setLevel(min(logging.INFO, log_level))

    # Setting the log format for the handlers
    log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s '
                                   '%(message)s')
    stream_handler.setFormatter(log_format)
    stream_handler_err.setFormatter(log_format)
    if file_handler is not None:
      file_handler.setFormatter(log_format)

    # Adding the handlers to the logger
    crappy_log.addHandler(stream_handler)
    crappy_log.addHandler(stream_handler_err)
    if file_handler is not None:
      crappy_log.addHandler(file_handler)

    cls.logger = crappy_log

  @classmethod
  def stop_all(cls) -> None:
    """Method for stopping all the Blocks by setting the stop event."""

    cls.stop_event.set()
    cls.logger.log(logging.INFO, 'Stop event set after a call to stop(), all '
                                 'Blocks should now finish')

  @classmethod
  def reset(cls) -> None:
    """Resets Crappy by emptying the WeakSet containing references to all the
    Blocks. Only useful for restarting Crappy from a script where Crappy was
    already started."""

    cls.instances = WeakSet()
    cls.logger.log(logging.INFO, 'Crappy was reset by the reset() command')

  @classmethod
  def _log_target(cls) -> None:
    """This method is the target to the logger Thread.

    It reads log messages from a Queue and passes them to the logger for
    handling.
    """

    while not cls.thread_stop:
      try:
        record = cls.log_queue.get(block=True, timeout=1)
      except Empty:
        continue

      logger = logging.getLogger(record.name)
      logger.handle(record)

  @staticmethod
  def _stdout_filter(rec: logging.LogRecord) -> bool:
    """Returns :obj:`True` if the input log message has level INFO or DEBUG,
    :obj:`False` otherwise."""

    return rec.levelno in (logging.DEBUG, logging.INFO)

  def run(self) -> None:
    """The method run by the Blocks when their process is started.

    It first calls :meth:`prepare`, then waits at the
    :obj:`multiprocessing.Barrier` for all Blocks to be ready, then calls
    :meth:`begin`, then :meth:`main`, and finally :meth:`finish`.
    
    If an exception is raised, sets the shared stop event to warn all the other
    Blocks.
    """

    try:
      # Initializes the logger for the Block
      self._set_block_logger()
      self._logger.log(logging.INFO, "Block launched")

      # Running the preliminary actions before the test starts
      try:
        self._logger.log(logging.INFO, "Block preparing")
        self.prepare()
      except (Exception,):
        # If exception is raised, breaking the barrier to warn the other blocks
        self._ready_barrier.abort()
        self._logger.log(logging.WARNING, "Breaking the barrier due to caught "
                                          "exception while preparing")
        raise

      # Waiting for all blocks to be ready, except if the barrier was broken
      try:
        self._logger.log(logging.INFO, "Waiting for the other Blocks to be "
                                       "ready")
        self._ready_barrier.wait()
        self._logger.log(logging.INFO, "All Blocks ready now")
      except BrokenBarrierError:
        raise PrepareError

      # Waiting for t0 to be set, should take a few milliseconds at most
      self._logger.log(logging.INFO, "Waiting for the start time to be set")
      self._start_event.wait(timeout=1)
      if not self._start_event.is_set():
        raise StartTimeout
      else:
        self._logger.log(logging.INFO, "Start time set, Block starting")

      # Running the first loop
      self._logger.log(logging.INFO, "Calling begin method")
      self.begin()

      # Setting the attributes for counting the performance
      self._last_t = time()
      self._last_fps = self._last_t
      self._n_loops = 0

      # Running the main loop until told to stop
      self._logger.log(logging.INFO, "Entering main loop")
      self.main()
      self._logger.log(logging.INFO, "Exiting main loop after stop event was "
                                     "set")

    # A wrong data type was sent through a Link
    except LinkDataError:
      self._logger.log(logging.ERROR, "Tried to send a wrong data type through"
                                      " a Link, stopping !")
    # An error occurred in another Block while preparing
    except PrepareError:
      self._logger.log(logging.ERROR, "Exception raised in another Block while"
                                      " waiting for all Blocks to be ready, "
                                      "stopping")
    # An error occurred in a Camera process while preparing
    except CameraPrepareError:
      self._logger.log(logging.ERROR, "Exception raised in a Camera process "
                                      "while preparing, stopping")
      # An error occurred in a Camera process while running
    except CameraRuntimeError:
      self._logger.log(logging.ERROR, "Exception raised in a Camera process "
                                      "while running, stopping")
    # The start event took too long to be set
    except StartTimeout:
      self._logger.log(logging.ERROR, "Waited too long for start time to be "
                                      "set, aborting !")
    # Tried to access t0 but it's not set yet
    except T0NotSetError:
      self._logger.log(logging.ERROR, "Trying to get the value of t0 when it's"
                                      " not set yet, aborting")
    # A Generator Block finished its path
    except GeneratorStop:
      self._logger.log(logging.WARNING, f"Generator path exhausted, stopping "
                                        f"the Block")
    # A File_reader Camera object has no more file to read from
    except ReaderStop:
      self._logger.log(logging.WARNING, "Exhausted all the images to read from"
                                        " a File_reader camera, stopping the "
                                        "Block")
    # The user requested the script to stop
    except KeyboardInterrupt:
      self._logger.log(logging.INFO, f"KeyBoardInterrupt caught, stopping")
    # Another exception occurred
    except (Exception,) as exc:
      logging.exception("Caught exception while running !", exc_info=exc)
      raise

    # In all cases, trying to properly close the block
    finally:
      self._logger.log(logging.INFO, "Setting the stop event")
      self._stop_event.set()
      self._logger.log(logging.INFO, "Calling the finish method")
      self.finish()

  def main(self) -> None:
    """The main loop of the :meth:`run` method. Repeatedly calls the
    :meth:`loop` method and manages the looping frequency."""

    # Looping until told to stop or an error occurs
    while not self._stop_event.is_set():
      self._logger.log(logging.DEBUG, "Looping")
      self.loop()
      self._logger.log(logging.DEBUG, "Handling freq")
      self._handle_freq()

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

    self._logger.log(logging.WARNING, f"[Block {type(self).__name__}] Loop "
                                      f"method not defined, this block does "
                                      f"nothing !")
    sleep(1)

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

    # Displaying frequency every 2 seconds
    if self.verbose and self._last_t - self._last_fps > 2:
      self._logger.log(
        logging.INFO,
        f"loops/s: {self._n_loops / (self._last_t - self._last_fps)}")

      self._n_loops = 0
      self._last_fps = self._last_t

  def _set_block_logger(self) -> None:
    """Initializes the logger for the Block.

    If the :mod:`multiprocessing` start method is `spawn` (mostly on Windows),
    redirects the log messages to a Queue for passing them to the main process.
    """

    log_level = 10 * int(round(self._log_level / 10, 0))

    logger = logging.getLogger(f'crappy.{self.name}')
    logger.setLevel(min(log_level, logging.INFO))

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn":
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(min(log_level, logging.INFO))
      logger.addHandler(queue_handler)

    self._logger = logger

  @property
  def t0(self) -> float:
    """Returns the value of t0, the exact moment when the test started that is
    shared between all the Blocks."""

    if self._instance_t0 is not None and self._instance_t0.value > 0:
      self._logger.log(logging.DEBUG, "Start time value requested")
      return self._instance_t0.value
    else:
      raise T0NotSetError

  def add_output(self, link: Link) -> None:
    """Adds an output link to the list of output links of the Block."""

    self.outputs.append(link)

  def add_input(self, link: Link) -> None:
    """Adds an input link to the list of input links of the Block."""

    self.inputs.append(link)

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the Block. This option should be
    preferred to calling :func:`print`.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(log_level, msg)

  def send(self, data: Union[Dict[str, Any], List[Any]]) -> None:
    """Ensures that the data to send is formatted as a :obj:`dict`, and sends
    it in all the downstream links."""

    # Building the dict to send from the data and labels if the data is a list
    if isinstance(data, list):
      if not self.labels:
        self._logger.log(logging.ERROR, "trying to send data as a list but no "
                                        "labels are specified ! Please add a "
                                        "self.labels attribute.")
        raise LinkDataError
      self._logger.log(logging.DEBUG, f"Converting {data} to dict before "
                                      f"sending")
      data = dict(zip(self.labels, data))

    # Making sure the data is being sent as a dict
    elif not isinstance(data, dict):
      self._logger.log(logging.ERROR, f"Trying to send a {type(data)} in a "
                                      f"Link !")
      raise LinkDataError

    # Sending the data to the downstream blocks
    for link in self.outputs:
      self._logger.log(logging.DEBUG, f"Sending {data} to Link {link}")
      link.send(data)

  def data_available(self) -> bool:
    """Returns :obj:`True` if there's data available for reading in at least
    one of the input Links."""

    self._logger.log(logging.DEBUG, "Data availability requested")
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

    self._logger.log(logging.DEBUG, f"Called recv_data, got {ret}")
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

    self._logger.log(logging.DEBUG, f"Called recv_last_data, got {ret}")
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
    self._logger.log(logging.DEBUG, f"Called recv_all_data, got {dict(ret)}")
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

    self._logger.log(logging.DEBUG, f"Called recv_all_data_raw, got "
                                    f"{[dict(dic) for dic in ret]}")
    return [dict(dic) for dic in ret]

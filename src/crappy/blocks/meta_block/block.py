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
from time import sleep, time, time_ns
from weakref import WeakSet
from typing import Union, Optional, List, Dict, Any
from collections import defaultdict
import subprocess
from sys import stdout, stderr, argv
from pathlib import Path

from .meta_block import MetaBlock
from ...links import Link
from ..._global import LinkDataError, StartTimeout, PrepareError, \
  T0NotSetError, GeneratorStop, ReaderStop, CameraPrepareError, \
  CameraRuntimeError, CameraConfigError
from ...tool.ft232h import USBServer

# TODO:
#   Increase granularity for the recv_all_data_raw method


class Block(Process, metaclass=MetaBlock):
  """This class constitutes the base object in Crappy.

  It is extremely versatile, an can perform a wide variety of actions during a
  test. Many Blocks are already defined in Crappy, but it is possible to define
  custom ones for specific purposes.

  It is a subclass of :obj:`multiprocessing.Process`, and is thus an
  independent process in Python. It communicates with other Blocks via
  :mod:`multiprocessing` objects.
  """

  instances = WeakSet()
  names: List[str] = list()
  log_level: Optional[int] = logging.DEBUG

  # The synchronization objects will be set later
  shared_t0: Optional[Synchronized] = None
  ready_barrier: Optional[synchronize.Barrier] = None
  start_event: Optional[synchronize.Event] = None
  stop_event: Optional[synchronize.Event] = None
  logger: Optional[logging.Logger] = None
  log_queue: Optional[queues.Queue] = None
  log_thread: Optional[Thread] = None
  thread_stop: bool = False

  prepared_all: bool = False
  launched_all: bool = False

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
    self.display_freq = False
    self.name = self.get_name(type(self).__name__)

    # The synchronization objects will be set later
    self._instance_t0: Optional[Synchronized] = None
    self._ready_barrier: Optional[synchronize.Barrier] = None
    self._start_event: Optional[synchronize.Event] = None
    self._stop_event: Optional[synchronize.Event] = None

    # The objects for logging will be set later
    self._log_queue: Optional[queues.Queue] = None
    self._logger: Optional[logging.Logger] = None
    self._debug: Optional[bool] = False
    self._log_level: int = logging.INFO

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
  def get_name(cls, name: str) -> str:
    """"""

    i = 1
    while f"crappy.{name}-{i}" in cls.names:
      i += 1

    cls.names.append(f"crappy.{name}-{i}")
    return f"crappy.{name}-{i}"

  @classmethod
  def start_all(cls,
                allow_root: bool = False,
                log_level: Optional[int] = logging.DEBUG) -> None:
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
      log_level: The maximum logging level that will be handled by Crappy. By
        default, it is set to the lowest level (DEBUG) so that all messages are
        handled. If set to a higher level, the levels specified for each Block
        with the ``debug`` argument may be ignored. If set to :obj:`None`,
        logging is totally disabled. Refer to the documentation of the
        :mod:`logging` module for information on the possible levels.
    """

    cls.prepare_all(log_level)
    cls.renice_all(allow_root)
    cls.launch_all()

  @classmethod
  def prepare_all(cls, log_level: Optional[int] = logging.DEBUG) -> None:
    """Creates the synchronization objects, shares them with the blocks, and
    starts the processes associated to the blocks.

    Also initializes the logger for the Crappy script.

    Once started with this method, the blocks will call their :meth:`prepare`
    method and then be blocked by a :obj:`multiprocessing.Barrier`.

    Args:
      log_level: The maximum logging level that will be handled by Crappy. By
        default, it is set to the lowest level (DEBUG) so that all messages are
        handled. If set to a higher level, the levels specified for each Block
        with the ``debug`` argument may be ignored. If set to :obj:`None`,
        logging is totally disabled. Refer to the documentation of the
        :mod:`logging` module for information on the possible levels.
    """

    try:

      if cls.prepared_all:
        cls.cls_log(logging.ERROR,
                    "The method prepare_all was already called ! Stop the "
                    "processes and reset Crappy before calling it again. Not "
                    "doing anything.")
        return
      if cls.launched_all:
        cls.cls_log(logging.ERROR,
                    "Please reset Crappy before calling the prepare_all "
                    "method again ! Not doing anything.")
        return

      cls.log_level = log_level

      # Initializing the logger and displaying the first messages
      cls._set_logger()
      cls.cls_log(logging.INFO,
                  "===================== CRAPPY =====================")
      cls.cls_log(logging.INFO, f'Starting the script {argv[0]}\n')
      cls.cls_log(logging.INFO, 'Logger configured')

      # Setting all the synchronization objects at the class level
      cls.ready_barrier = Barrier(len(cls.instances) + 1)
      cls.shared_t0 = Value('d', -1.0)
      cls.start_event = Event()
      cls.stop_event = Event()
      cls.cls_log(logging.INFO, 'Multiprocessing synchronization objects set '
                                'for main process')

      # Initializing the objects required for logging
      cls.log_queue = Queue()
      cls.log_thread = Thread(target=cls._log_target)
      if get_start_method() == 'spawn':
        cls.log_thread.start()
        cls.cls_log(logging.INFO, 'Logger thread started')

      # Starting the USB server if required
      if USBServer.initialized:
        cls.cls_log(logging.INFO, "Starting the USB server")
        USBServer.start_server(cls.log_queue, logging.INFO)

      # Passing the synchronization and logging objects to each block
      for instance in cls.instances:
        instance._ready_barrier = cls.ready_barrier
        instance._instance_t0 = cls.shared_t0
        instance._stop_event = cls.stop_event
        instance._start_event = cls.start_event
        instance._log_queue = cls.log_queue
        cls.cls_log(logging.INFO, f'Multiprocessing synchronization objects '
                                  f'set for {instance.name} Block')

        # Setting the common log level to all the instances
        if instance._log_level is not None:
          if cls.log_level is not None:
            instance._log_level = max(instance._log_level, cls.log_level)
          else:
            instance._log_level = None
        cls.cls_log(logging.INFO, f"Log level set for the {instance.name} "
                                  f"Block")

      # Starting all the blocks
      for instance in cls.instances:
        instance.start()
        cls.cls_log(logging.INFO, f'Started the {instance.name} Block')

      # Setting the prepared flag
      cls.prepared_all = True

    except KeyboardInterrupt:
      cls.cls_log(logging.INFO, 'KeyboardInterrupt caught while running '
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
      # Making sure the prepare method has already been called
      if not cls.prepared_all:
        cls.cls_log(logging.ERROR, "Cannot call renice before calling "
                                   "prepare ! Aborting")
        return
      if cls.launched_all:
        cls.cls_log(logging.ERROR,
                    "Please reset Crappy before calling the renice_all method "
                    "again ! Not doing anything.")

      # There's no niceness on Windows
      if system() == "Windows":
        cls.cls_log(logging.INFO, 'Not renicing processes on Windows')
        return

      # Renicing all the blocks
      cls.cls_log(logging.INFO, 'Renicing processes')
      for inst in cls.instances:
        # If root is not allowed then the minimum niceness is 0
        niceness = max(inst.niceness, 0 if not allow_root else -20)

        # System call for setting the niceness
        if niceness < 0:
          subprocess.call(['sudo', 'renice', str(niceness), '-p',
                           str(inst.pid)], stdout=subprocess.DEVNULL)
          cls.cls_log(logging.INFO, f"Reniced process {inst.name} with PID "
                                    f"{inst.pid} to niceness {niceness} "
                                    f"with sudo privilege")
        else:
          subprocess.call(['renice', str(niceness), '-p', str(inst.pid)],
                          stdout=subprocess.DEVNULL)
          cls.cls_log(logging.INFO, f"Reniced process {inst.name} with PID "
                                    f"{inst.pid} to niceness {niceness}")

    except KeyboardInterrupt:
      cls.cls_log(logging.INFO, 'KeyboardInterrupt caught while running '
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
      # Making sure the prepare method has already been called
      if not cls.prepared_all:
        cls.cls_log(logging.ERROR, "Cannot call launch_all before calling "
                                   "prepare ! Aborting")
        return
      if cls.launched_all:
        cls.cls_log(logging.ERROR,
                    "Please reset Crappy before calling the launch_all method "
                    "again ! Not doing anything.")
        return

      cls.launched_all = True

      # The barrier waits for the main process to be ready so that the
      # prepare_all and launch_all methods can be used separately for a finer
      # grained control
      cls.cls_log(logging.INFO, 'Waiting for all Blocks to be ready')
      cls.ready_barrier.wait()
      cls.cls_log(logging.INFO, 'All Blocks ready now')

      # Setting t0 and telling all the block to start
      cls.shared_t0.value = time_ns() / 1e9
      cls.cls_log(logging.INFO, f'Start time set to {cls.shared_t0.value}s')
      cls.start_event.set()
      cls.cls_log(logging.INFO, 'Start event set, all Blocks can now start')

      # The main process mustn't finish before all the blocks are done running
      cls.cls_log(logging.INFO, 'Main process done, waiting for all Blocks to '
                                'finish')
      for inst in cls.instances:
        inst.join()
        cls.cls_log(logging.INFO, f'{inst.name} finished by itself')

    # A Block crashed while preparing
    except BrokenBarrierError:
      cls.cls_log(logging.ERROR, "Exception raised in a Block while waiting "
                                 "for all Blocks to be ready, stopping")
      cls._exception()
    # The user ended the script while preparing
    except KeyboardInterrupt:
      cls.cls_log(logging.INFO, 'KeyboardInterrupt caught while running '
                                'launch_all')
      cls._exception()
    # Another exception occurred
    except (Exception,) as exc:
      cls.logger.exception("Caught exception while running launch_all, "
                           "aborting", exc_info=exc)
      cls._exception()

    # Performing the cleanup actions before exiting
    finally:
      cls._cleanup()

  @classmethod
  def _exception(cls) -> None:
    """This method is called when an exception is caught in the main process.

    It waits for all the Blocks to end, and kills them if they don't stop by
    themselves. Also stops the thread managing the logging.
    """

    cls.stop_event.set()
    cls.cls_log(logging.INFO, 'Stop event set, waiting for all Blocks to '
                              'finish')
    t = time()

    # Waiting at most 3 seconds for all the blocks to finish
    while cls.instances and not all(not inst.is_alive() for inst
                                    in cls.instances):
      cls.cls_log(logging.INFO, "All Blocks not stopped yet")
      sleep(0.5)

      # After 3 seconds, killing the blocks that didn't stop
      if time() - t > 3:
        cls.cls_log(logging.WARNING, 'All Blocks not stopped, terminating the '
                                     'living ones')
        for inst in cls.instances:
          if inst.is_alive():
            inst.terminate()
            cls.cls_log(logging.WARNING, f'Block {inst.name} terminated')
          else:
            cls.cls_log(logging.INFO, f'Block {inst.name} done')

        break

  @classmethod
  def _cleanup(cls) -> None:
    """Method called at the very end of every script execution.

    It stops, if relevant, the USBServer and the log_thread, and warns the user
    in case processes would still be running.
    """

    try:
      # Stopping the USB server if required
      if USBServer.initialized:
        cls.cls_log(logging.INFO, "Stopping the USB server")
        USBServer.stop_server()

      # Stopping the log thread if required
      if get_start_method() == 'spawn':
        cls.thread_stop = True
        cls.log_thread.join(timeout=0.1)

      # Warning in case the log thread did not stop correctly
      if cls.log_thread.is_alive():
        cls.cls_log(logging.WARNING, "The thread reading the log messages did "
                                     "not terminate in time !")

      # Checking whether all Blocks terminated gracefully
      if cls.instances and any(inst.is_alive() for inst in cls.instances):
        running = ', '.join(inst.name for inst in cls.instances
                            if inst.is_alive())
        cls.cls_log(logging.ERROR, f"Crappy failed to finish gracefully, "
                                   f"Block(s) {running} still running !")
      else:
        cls.cls_log(logging.INFO, 'All Blocks done, Crappy terminated '
                                  'gracefully\n')

    except KeyboardInterrupt:
      cls.cls_log(logging.INFO, "Caught KeyboardInterrupt while cleaning up, "
                                "ignoring it")
    except (Exception,) as exc:
      cls.logger.exception("Caught exception while cleaning up !",
                           exc_info=exc)

  @classmethod
  def _set_logger(cls) -> None:
    """Initializes the logging for the main process.

    It creates two Stream Loggers, one for the info and debug levels displaying
    on stdout and one for the other levels displaying on stderr. It also
    creates a File Logger for saving the log to a log file.

    The levels WARNING and above are always being displayed in the terminal, no
    matter what the user chooses. Similarly, the INFO log level and above are
    always being saved to the log file.
    """

    # The logger handling all messages
    crappy_log = logging.getLogger('crappy')

    if cls.log_level is not None:
      crappy_log.setLevel(cls.log_level)
    else:
      logging.disable()

    # In case there's no logging, no need to configure the handlers
    if cls.log_level is not None:
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
        file_handler = logging.FileHandler(log_path / 'logs.txt', mode='w')
      else:
        file_handler = None

      # Setting the log levels for the handlers
      stream_handler.setLevel(max(logging.DEBUG, cls.log_level))
      stream_handler.addFilter(cls._stdout_filter)
      stream_handler_err.setLevel(max(logging.WARNING, cls.log_level))
      if file_handler is not None:
        file_handler.setLevel(max(logging.DEBUG, cls.log_level))

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

    if cls.stop_event is not None:
      cls.stop_event.set()
      cls.cls_log(logging.INFO, 'Stop event set after a call to stop(), all '
                                'Blocks should now finish')

  @classmethod
  def reset(cls) -> None:
    """Resets Crappy by emptying the WeakSet containing references to all the
    Blocks. Only useful for restarting Crappy from a script where Crappy was
    already started."""

    cls.instances = WeakSet()
    cls.names = list()
    cls.thread_stop = False
    cls.prepared_all = False
    cls.launched_all = False

    cls.shared_t0 = None
    cls.ready_barrier = None
    cls.start_event = None
    cls.stop_event = None

    if cls.logger is not None:
      cls.cls_log(logging.INFO, 'Crappy was reset by the reset() command')
  
  @classmethod
  def cls_log(cls, level: int, msg: str) -> None:
    """Wrapper for logging messages in the main process.
    
    Ensures the logger exists before trying to log, thus avoiding potential 
    errors.
    """
    
    if cls.logger is None:
      return
    cls.logger.log(level=level, msg=msg)

  @classmethod
  def _log_target(cls) -> None:
    """This method is the target to the logger Thread.

    It reads log messages from a Queue and passes them to the logger for
    handling.
    """

    while not cls.thread_stop:
      try:
        record = cls.log_queue.get(block=True, timeout=0.05)
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
      self.log(logging.INFO, "Block launched")

      # Running the preliminary actions before the test starts
      try:
        self.log(logging.INFO, "Block preparing")
        self.prepare()
      except (Exception,):
        # If exception is raised, breaking the barrier to warn the other blocks
        self._ready_barrier.abort()
        self.log(logging.WARNING, "Breaking the barrier due to caught "
                                  "exception while preparing")
        raise

      # Waiting for all blocks to be ready, except if the barrier was broken
      try:
        self.log(logging.INFO, "Waiting for the other Blocks to be ready")
        self._ready_barrier.wait()
        self.log(logging.INFO, "All Blocks ready now")
      except BrokenBarrierError:
        raise PrepareError

      # Waiting for t0 to be set, should take a few milliseconds at most
      self.log(logging.INFO, "Waiting for the start time to be set")
      self._start_event.wait(timeout=1)
      if not self._start_event.is_set():
        raise StartTimeout
      else:
        self.log(logging.INFO, "Start time set, Block starting")

      # Running the first loop
      self.log(logging.INFO, "Calling begin method")
      self.begin()

      # Setting the attributes for counting the performance
      self._last_t = time_ns() / 1e9
      self._last_fps = self._last_t
      self._n_loops = 0

      # Running the main loop until told to stop
      self.log(logging.INFO, "Entering main loop")
      self.main()
      self.log(logging.INFO, "Exiting main loop after stop event was set")

    # A wrong data type was sent through a Link
    except LinkDataError:
      self.log(logging.ERROR, "Tried to send a wrong data type through a Link,"
                              " stopping !")
    # An error occurred in another Block while preparing
    except PrepareError:
      self.log(logging.ERROR, "Exception raised in another Block while waiting"
                              " for all Blocks to be ready, stopping")
    # An error occurred in the CameraConfig window while preparing
    except CameraConfigError:
      self.log(logging.ERROR, "Exception raised in a configuration window, "
                              "stopping")
    # An error occurred in a Camera process while preparing
    except CameraPrepareError:
      self.log(logging.ERROR, "Exception raised in a Camera process while "
                              "preparing, stopping")
      # An error occurred in a Camera process while running
    except CameraRuntimeError:
      self.log(logging.ERROR, "Exception raised in a Camera process while "
                              "running, stopping")
    # The start event took too long to be set
    except StartTimeout:
      self.log(logging.ERROR, "Waited too long for start time to be set, "
                              "aborting !")
    # Tried to access t0 but it's not set yet
    except T0NotSetError:
      self.log(logging.ERROR, "Trying to get the value of t0 when it's not "
                              "set yet, aborting")
    # A Generator Block finished its path
    except GeneratorStop:
      self.log(logging.WARNING, f"Generator path exhausted, stopping the "
                                f"Block")
    # A FileReader Camera object has no more file to read from
    except ReaderStop:
      self.log(logging.WARNING, "Exhausted all the images to read from a "
                                "FileReader camera, stopping the Block")
    # The user requested the script to stop
    except KeyboardInterrupt:
      self.log(logging.INFO, f"KeyBoardInterrupt caught, stopping")
    # Another exception occurred
    except (Exception,) as exc:
      self._logger.exception("Caught exception while running !", exc_info=exc)

    # In all cases, trying to properly close the block
    finally:
      self.log(logging.INFO, "Setting the stop event")
      self._stop_event.set()
      self.log(logging.INFO, "Calling the finish method")
      try:
        self.finish()
      except KeyboardInterrupt:
        self.log(logging.INFO, "Caught KeyboardInterrupt while finishing, "
                               "ignoring it")
      except (Exception,) as exc:
        self._logger.exception("Caught exception while finishing !",
                               exc_info=exc)

  def main(self) -> None:
    """The main loop of the :meth:`run` method. Repeatedly calls the
    :meth:`loop` method and manages the looping frequency."""

    # Looping until told to stop or an error occurs
    while not self._stop_event.is_set():
      self.log(logging.DEBUG, "Looping")
      self.loop()
      self.log(logging.DEBUG, "Handling freq")
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

    self.log(logging.WARNING, f"Loop method not defined, this block does "
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

  def stop(self) -> None:
    """This method stops all the running Blocks.

    It should be called from the :meth:`loop` method of a Block. It allows to
    stop the execution of the script in a clean way, without raising an
    exception. It is mostly intended for users writing their own Blocks.

    Note:
      Calling this method in :meth:`__init__`, :meth:`prepare` or :meth:`begin`
      is not recommended, as the Block will only stop when reaching the
      :meth:`loop` method. Calling this method during :meth:`finish` will have
      no effect.
    """

    if self._stop_event is not None:
      self.log(logging.WARNING, "stop method called, setting the stop event !")
      self._stop_event.set()

  def _handle_freq(self) -> None:
    """This method ensures that the Block loops at the desired frequency, or as
    fast as possible if the requested frequency cannot be achieved.

    It also displays the looping frequency of the Block if requested by the
    user. If no looping frequency is specified, the Block will loop as fast as
    possible.
    """

    self._n_loops += 1
    t = time_ns() / 1e9

    # Only handling frequency if requested
    if self.freq is not None:

      # Correcting the error of the sleep function through a recursive approach
      # The last 2 milliseconds are in free loop
      remaining = self._last_t + 1 / self.freq - t
      while remaining > 0:
        t = time_ns() / 1e9
        remaining = self._last_t + 1 / self.freq - t
        sleep(max(0., remaining / 2 - 2e-3))

    self._last_t = t

    # Displaying frequency every 2 seconds
    if self.display_freq and self._last_t - self._last_fps > 2:
      self.log(
        logging.INFO,
        f"loops/s: {self._n_loops / (self._last_t - self._last_fps)}")

      self._n_loops = 0
      self._last_fps = self._last_t

  def _set_block_logger(self) -> None:
    """Initializes the logger for the Block.

    If the :mod:`multiprocessing` start method is `spawn` (mostly on Windows),
    redirects the log messages to a Queue for passing them to the main process.
    """

    logger = logging.getLogger(self.name)

    # Adjusting logging to the desired level
    if self._log_level is not None:
      logger.setLevel(self._log_level)
    else:
      logging.disable()

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn" and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger

  @property
  def debug(self) -> Optional[bool]:
    """Indicates whether the debug information should be displayed or not.

    If :obj:`False` (the default), only displays the INFO logging level. If
    :obj:`True`, displays the DEBUG logging level for the Block. And if
    :obj:`None`, displays only the CRITICAL logging level, which is equivalent
    to no information at all.
    """

    return self._debug

  @debug.setter
  def debug(self, val: Optional[bool]) -> None:
    if val is not None:
      if val:
        self._debug = True
        self._log_level = logging.DEBUG
      else:
        self._debug = False
        self._log_level = logging.INFO
    else:
      self._debug = None
      self._log_level = logging.CRITICAL

  @property
  def t0(self) -> float:
    """Returns the value of t0, the exact moment when the test started that is
    shared between all the Blocks."""

    if self._instance_t0 is not None and self._instance_t0.value > 0:
      self.log(logging.DEBUG, "Start time value requested")
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
        self.log(logging.ERROR, "trying to send data as a list but no labels "
                                "are specified ! Please add a self.labels "
                                "attribute.")
        raise LinkDataError
      self.log(logging.DEBUG, f"Converting {data} to dict before sending")
      data = dict(zip(self.labels, data))

    # Making sure the data is being sent as a dict
    elif not isinstance(data, dict):
      self.log(logging.ERROR, f"Trying to send a {type(data)} in a Link !")
      raise LinkDataError

    # Sending the data to the downstream blocks
    for link in self.outputs:
      self.log(logging.DEBUG, f"Sending {data} to Link {link.name}")
      link.send(data)

  def data_available(self) -> bool:
    """Returns :obj:`True` if there's data available for reading in at least
    one of the input Links."""

    self.log(logging.DEBUG, "Data availability requested")
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

    self.log(logging.DEBUG, f"Called recv_data, got {ret}")
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

    self.log(logging.DEBUG, f"Called recv_last_data, got {ret}")
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
    self.log(logging.DEBUG, f"Called recv_all_data, got {dict(ret)}")
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

    self.log(logging.DEBUG, f"Called recv_all_data_raw, got "
                            f"{[dict(dic) for dic in ret]}")
    return [dict(dic) for dic in ret]

# coding: utf-8

from platform import system
from multiprocessing import Process, Value, Barrier, Event, Queue, \
  get_start_method, synchronize, queues
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.connection import wait
from threading import BrokenBarrierError, Thread
from queue import Empty
import logging
import logging.handlers
from time import sleep, time, time_ns
from weakref import WeakSet
from typing import Union, Optional, Any
from collections.abc import Iterable
from collections import defaultdict
import subprocess
from sys import stdout, stderr, argv
from pathlib import Path

from .meta_block import MetaBlock
from ...links import Link
from ..._global import LinkDataError, StartTimeout, PrepareError, \
  T0NotSetError, GeneratorStop, ReaderStop, CameraPrepareError, \
  CameraRuntimeError, CameraConfigError, CrappyFail
from ...tool.ft232h import USBServer


class Block(Process, metaclass=MetaBlock):
  """This class constitutes the base object in Crappy.

  It is extremely versatile, and can perform a wide variety of actions during a
  test. Many Blocks are already defined in Crappy, but it is possible to define
  custom ones for specific purposes.

  It is a subclass of :obj:`multiprocessing.Process`, and is thus an
  independent process in Python. It communicates with other Blocks via
  :mod:`multiprocessing` objects.

  This class also contains the class methods that allow driving a script with
  Crappy. They are always called in the `__main__` Process, and drive the
  execution of all the children Blocks.
  
  .. versionadded:: 1.4.0
  """

  instances = WeakSet()
  names: list[str] = list()
  log_level: Optional[int] = logging.DEBUG

  # The synchronization objects will be set later
  shared_t0: Optional[Synchronized] = None
  ready_barrier: Optional[synchronize.Barrier] = None
  start_event: Optional[synchronize.Event] = None
  pause_event: Optional[synchronize.Event] = None
  stop_event: Optional[synchronize.Event] = None
  raise_event: Optional[synchronize.Event] = None
  kbi_event: Optional[synchronize.Event] = None
  logger: Optional[logging.Logger] = None
  log_queue: Optional[queues.Queue] = None
  log_thread: Optional[Thread] = None
  thread_stop: bool = False
  no_raise: bool = False

  prepared_all: bool = False
  launched_all: bool = False

  def __init__(self) -> None:
    """Sets the attributes and initializes the parent class."""

    super().__init__()

    # The lists of input and output links
    self.outputs: list[Link] = list()
    self.inputs: list[Link] = list()

    # Various objects that should be set by child classes
    self.niceness: int = 0
    self.labels: Optional[Iterable[str]] = None
    self.freq: Optional[float] = None
    self.display_freq: bool = False
    self.name: str = self.get_name(type(self).__name__)
    self.pausable: bool = True

    # The synchronization objects will be set later
    self._instance_t0: Optional[Synchronized] = None
    self._ready_barrier: Optional[synchronize.Barrier] = None
    self._start_event: Optional[synchronize.Event] = None
    self._pause_event: Optional[synchronize.Event] = None
    self._stop_event: Optional[synchronize.Event] = None
    self._raise_event: Optional[synchronize.Event] = None
    self._kbi_event: Optional[synchronize.Event] = None

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
    """Called when instantiating a new instance of a Block.

    Adds itself to the :obj:`~weakref.WeakSet` listing all the instantiated
    Blocks.
    """

    instance = super().__new__(cls)
    cls.instances.add(instance)
    return instance

  @classmethod
  def get_name(cls, name: str) -> str:
    """Method attributing to each new Block a unique name, based on the name of
    the class and the number of existing instances for this class.
    
    .. versionadded:: 2.0.0
    """

    i = 1
    while f"crappy.{name}-{i}" in cls.names:
      i += 1

    cls.names.append(f"crappy.{name}-{i}")
    return f"crappy.{name}-{i}"

  @classmethod
  def start_all(cls,
                allow_root: bool = False,
                log_level: Optional[int] = logging.DEBUG,
                no_raise: bool = False) -> None:
    """Method for starting a script with Crappy.

    It sets the synchronization objects for all the Blocks, renices the
    corresponding :obj:`~multiprocessing.Process` and starts the Blocks.

    The call to this method is blocking until Crappy finishes.

    Note:
      It is possible to have a finer grained control of the start of a Crappy
      script with the methods :meth:`~crappy.blocks.Block.prepare_all`,
      :meth:`~crappy.blocks.Block.renice_all` and
      :meth:`~crappy.blocks.Block.launch_all`.

    Args:
      allow_root: If set to :obj:`True`, tries to renice the Processes with
        sudo privilege in Linux. It requires the Python script to be run
        with sudo privilege, otherwise it has no effect.

        .. versionchanged:: 2.0.0 renamed from *high_prio* to *allow_root*
      log_level: The maximum logging level that will be handled by Crappy. By
        default, it is set to the lowest level (:obj:`~logging.DEBUG`) so that
        all messages are handled. If set to a higher level, the levels
        specified for each Block with the ``debug`` argument may be ignored. If
        set to :obj:`None`, logging is totally disabled. Refer to the
        documentation of the :mod:`logging` module for information on the
        possible levels.

        .. versionadded:: 2.0.0
      no_raise: When set to :obj:`False`, the Exceptions encountered during
        Crappy's execution, as well as the :exc:`KeyboardInterrupt`, will raise
        an Exception right before Crappy returns. This is meant to prevent the
        execution of code that would come after Crappy, in case Crappy does not
        terminate as expected. This behavior can be disabled by setting this
        argument to :obj:`True`.

        .. versionadded:: 2.0.0

    .. versionremoved:: 2.0.0 *t0*, *verbose*, *bg* arguments
    """

    cls.prepare_all(log_level)
    cls.renice_all(allow_root)
    cls.launch_all(no_raise)

  @classmethod
  def prepare_all(cls, log_level: Optional[int] = logging.DEBUG) -> None:
    """Creates the synchronization objects, shares them with the Blocks, and
    starts the :obj:`~multiprocessing.Process` associated to the Blocks.

    Also initializes the :obj:`~logging.Logger` for the Crappy script.

    Once started with this method, the Blocks will call their
    :meth:`~crappy.blocks.Block.prepare` method and then be blocked by a
    :obj:`multiprocessing.Barrier`.

    If an error is caught at a moment when the Blocks might already be running,
    performs an extensive cleanup to ensure everything stops as expected.

    Args:
      log_level: The maximum logging level that will be handled by Crappy. By
        default, it is set to the lowest level (:obj:`~logging.DEBUG`) so that 
        all messages are handled. If set to a higher level, the levels 
        specified for each Block with the ``debug`` argument may be ignored. If 
        set to :obj:`None`, logging is totally disabled. Refer to the 
        documentation of the :mod:`logging` module for information on the 
        possible levels.

        .. versionadded:: 2.0.0
    
    .. versionremoved:: 2.0.0 *verbose* argument
    """

    # Flag indicating whether to perform the cleanup action or not
    cleanup = False

    try:
      # Making sure that the Block classmethods are called in the right order
      if cls.prepared_all:
        cls.cls_log(logging.ERROR,
                    "The method prepare_all was already called ! This is "
                    "unexpected, aborting !")
        # As Crappy was already initialized, it must now be cleaned up
        cleanup = True
        # Raising will skip all the setup part and keep the existing context
        raise RuntimeError

      if cls.launched_all:
        cls.cls_log(logging.ERROR, "The launched_all flag is unexpectedly "
                                   "raised, aborting !")
        # As Crappy was already initialized, it must now be cleaned up
        cleanup = True
        # Raising will skip all the setup part and keep the existing context
        raise RuntimeError

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
      cls.pause_event = Event()
      cls.stop_event = Event()
      cls.raise_event = Event()
      cls.kbi_event = Event()
      cls.cls_log(logging.INFO, 'Multiprocessing synchronization objects set '
                                'for main process')

      # Starting from that point, Crappy has to be cleaned up if anything wrong
      # happens
      cleanup = True

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

      # Passing the synchronization and logging objects to each Block
      for instance in cls.instances:
        instance._ready_barrier = cls.ready_barrier
        instance._instance_t0 = cls.shared_t0
        instance._stop_event = cls.stop_event
        instance._pause_event = cls.pause_event
        instance._start_event = cls.start_event
        instance._raise_event = cls.raise_event
        instance._kbi_event = cls.kbi_event
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

      # Starting all the Blocks
      for instance in cls.instances:
        instance.start()
        cls.cls_log(logging.INFO, f'Started the {instance.name} Block')

      # Setting the prepared flag
      cls.prepared_all = True

    # At that point the Blocks might be started or not. If started, they are
    # preparing or waiting at the Barrier
    except (Exception, KeyboardInterrupt) as exc:

      # If there is no specific cleanup to perform, only raising
      if not cleanup:
        raise

      # KeyboardInterrupt is a separate case
      if isinstance(exc, KeyboardInterrupt):
        cls.cls_log(logging.WARNING, "Caught KeyboardInterrupt in the main "
                                     "Process while running prepare_all !")
        # Special Event for the KeyboardInterrupt
        cls.kbi_event.set()
        cls.cls_log(logging.WARNING, 'Set the KbI Event after catching '
                                     'KeyboardInterrupt in the main Process '
                                     'in prepare_all')
      # General case
      else:
        cls.logger.exception("Caught exception while running prepare_all, "
                             "aborting", exc_info=exc)
        # Any Exception caught in the main Process must stop the script
        cls.raise_event.set()
        cls.cls_log(logging.WARNING, 'Set the raise Event after exception was '
                                     'caught in the main Process in '
                                     'prepare_all')
      # Breaking the Barrier to warn other Processes that something went wrong
      cls.ready_barrier.abort()
      cls.cls_log(logging.WARNING, "Broke the Barrier due to an exception "
                                   "caught in prepare_all")
      # Need to clean up as some Blocks might already be running
      cls._cleanup()

  @classmethod
  def renice_all(cls, allow_root: bool) -> None:
    """On Linux and macOS, renices the :obj:`~multiprocessing.Process` 
    associated with the Blocks.

    On Windows, does nothing.

    If an error is caught, performs an extensive cleanup to ensure everything
    stops as expected.

    Args:
      allow_root: If set to :obj:`True`, tries to renice the Processes with 
        sudo privilege in Linux. It requires the Python script to be run with 
        sudo privilege, otherwise it has no effect.

        .. versionchanged:: 2.0.0 renamed from *high_prio* to *allow_root*
    """

    # Flag indicating whether to perform the cleanup action or not
    cleanup = True

    try:
      # Making sure that the Block classmethods are called in the right order
      if not cls.prepared_all:
        cls.cls_log(logging.ERROR, "Cannot call renice before calling "
                                   "prepare ! Aborting")
        # If prepare wasn't called, there is no need to clean up Crappy
        cleanup = False
        raise RuntimeError("Cannot call renice before calling prepare ! "
                           "Aborting")

      if cls.launched_all:
        cls.cls_log(logging.ERROR, "The launched_all flag is unexpectedly "
                                   "raised, aborting !")
        raise RuntimeError

      # There's no niceness on Windows
      if system() == "Windows":
        cls.cls_log(logging.INFO, 'Not renicing processes on Windows')
        return

      # Renicing all the Blocks
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

    # At that point the Blocks should be preparing or waiting at the Barrier
    except (Exception, KeyboardInterrupt) as exc:

      # If there is no specific cleanup to perform, only raising
      if not cleanup:
        raise

      # KeyboardInterrupt is a separate case
      if isinstance(exc, KeyboardInterrupt):
        cls.cls_log(logging.WARNING, "Caught KeyboardInterrupt in the main "
                                     "Process while running renice_all !")
        # Special Event for the KeyboardInterrupt
        cls.kbi_event.set()
        cls.cls_log(logging.WARNING, 'Set the KbI Event after catching '
                                     'KeyboardInterrupt in the main Process '
                                     'in renice_all')
      # General case
      else:
        cls.logger.exception("Caught exception while running renice_all, "
                             "aborting", exc_info=exc)
        # Any Exception caught in the main Process must stop the script
        cls.raise_event.set()
        cls.cls_log(logging.WARNING, 'Set the raise Event after exception was '
                                     'caught in the main Process in '
                                     'renice_all')
      # Breaking the Barrier to warn other Processes that something went wrong
      cls.ready_barrier.abort()
      cls.cls_log(logging.WARNING, "Broke the Barrier due to an exception "
                                   "caught in renice_all")
      # Need to clean up the running Blocks and other Processes / Threads
      cls._cleanup()

  @classmethod
  def launch_all(cls, no_raise: bool = False) -> None:
    """The final method being called by the main
    :obj:`~multiprocessing.Process` running a script with Crappy.

    It unlocks all the Blocks by releasing the synchronization
    :obj:`~multiprocessing.Barrier`, sets the shared t0
    :obj:`~multiprocessing.Value`, and then waits for all the Blocks to finish.

    In case an exception is raised, sets the stop :obj:`~multiprocessing.Event`
    for warning the Blocks, waits for the Blocks to finish, and if they don't,
    terminates them.

    Args:
      no_raise: When set to :obj:`False`, the Exceptions encountered during
        Crappy's execution, as well as the :exc:`KeyboardInterrupt`, will raise
        an Exception right before Crappy returns. This is meant to prevent the
        execution of code that would come after Crappy, in case Crappy does not
        terminate as expected. This behavior can be disabled by setting this
        argument to :obj:`True`.
    
    .. versionremoved:: 2.0.0 *t0*, *verbose* and *bg* arguments
    """

    # Setting the no_raise flag
    cls.no_raise = no_raise

    # Flag indicating whether to perform the cleanup action or not
    cleanup = True

    try:
      # Making sure that the Block classmethods are called in the right order
      if not cls.prepared_all:
        cls.cls_log(logging.ERROR, "Cannot call launch_all before calling "
                                   "prepare_all ! Aborting")
        # If prepare wasn't called, there is no need to clean up Crappy
        cleanup = False
        raise RuntimeError("Cannot call launch before calling prepare ! "
                           "Aborting")

      if cls.launched_all:
        cls.cls_log(logging.ERROR, "The launched_all flag is unexpectedly "
                                   "raised, aborting !")
        raise RuntimeError

      cls.launched_all = True

      # The Barrier waits for the main Process to be ready so that the
      # prepare_all and launch_all methods can be used separately for a finer
      # grained control
      cls.cls_log(logging.INFO, 'Waiting for all Blocks to be ready')
      cls.ready_barrier.wait()
      cls.cls_log(logging.INFO, 'All Blocks ready now')

      # Setting t0 and telling all the Blocks to start
      cls.shared_t0.value = time_ns() / 1e9
      cls.cls_log(logging.INFO, f'Start time set to {cls.shared_t0.value}s')
      cls.start_event.set()
      cls.cls_log(logging.INFO, 'Start event set, all Blocks can now start')

      # The main Process mustn't finish before all the Blocks are stopped
      cls.cls_log(logging.INFO, 'Main Process done, waiting for all Blocks to '
                                'finish')
      for _ in wait([inst.sentinel for inst in cls.instances]):
        cls.cls_log(logging.INFO, "A Block has finished, waiting for the "
                                  "other ones to follow")

    except (BrokenBarrierError, KeyboardInterrupt, Exception) as exc:

      # If there is no specific cleanup to perform, only raising
      if not cleanup:
        raise

      # KeyboardInterrupt is a separate case
      if isinstance(exc, KeyboardInterrupt):
        cls.cls_log(logging.WARNING, "Caught KeyboardInterrupt in the main "
                                     "Process while running launch_all !")
        # Special Event for the KeyboardInterrupt
        cls.kbi_event.set()
        cls.cls_log(logging.WARNING, 'Set the KbI Event after catching '
                                     'KeyboardInterrupt in the main Process '
                                     'in launch_all')
      # Case when a Block crashed while preparing
      elif isinstance(exc, BrokenBarrierError):
        cls.cls_log(logging.ERROR, "Exception raised in a Block while waiting "
                                   "for all Blocks to be ready, stopping")
      # General case
      else:
        cls.logger.exception("Caught exception while running launch_all, "
                             "aborting", exc_info=exc)
        # Any Exception caught in the main Process must stop the script
        cls.raise_event.set()
        cls.cls_log(logging.WARNING, 'Set the raise Event after exception was '
                                     'caught in the main Process in '
                                     'launch_all')
      # Breaking the Barrier to warn other Processes that something went wrong
      cls.ready_barrier.abort()
      cls.cls_log(logging.WARNING, "Broke the Barrier due to an exception "
                                   "caught in launch_all")
    finally:
      # Need to clean up the running Blocks and other Processes / Threads
      if cleanup:
        cls._cleanup()

  @classmethod
  def _cleanup(cls) -> None:
    """Method called at the very end of every script execution.

    It first waits for all the Blocks to end, and kills them if they don't stop
    by themselves. Then, it also stops, if relevant, the USBServer and the
    log_thread, and warns the user in case Processes would still be running.

    Finally, it raises an exception if needed, in order to stop the script of
    the main Process. This way, any action that could follow the normal
    execution of Crappy won't happen, unless the user explicitly catches
    Crappy's exception and decides to go on with the script.
    """

    try:

      # Setting the stop Event, to indicate all the Blocks to finish
      cls.stop_event.set()
      cls.cls_log(logging.INFO, 'Stop event set, waiting for all Blocks to '
                                'finish')
      t = time()

      # Waiting at most 3 seconds for all the Blocks to finish
      while cls.instances and not all(not inst.is_alive() for inst
                                      in cls.instances):
        cls.cls_log(logging.INFO, "All Blocks not stopped yet")
        sleep(0.5)

        # After 3 seconds, killing the Blocks that didn't stop
        if time() - t > 3:
          cls.cls_log(logging.WARNING, 'All Blocks not stopped, terminating '
                                       'the living ones')
          for inst in cls.instances:
            if inst.is_alive():
              inst.terminate()
              cls.cls_log(logging.WARNING, f'Block {inst.name} terminated')
            else:
              cls.cls_log(logging.INFO, f'Block {inst.name} done')

          break

      # Stopping the USB server if required
      if USBServer.initialized:
        cls.cls_log(logging.INFO, "Stopping the USB server")
        USBServer.stop_server()

      # Stopping the log thread if required
      if get_start_method() == 'spawn' and cls.log_thread is not None:
        cls.thread_stop = True
        cls.log_thread.join(timeout=0.1)

      # Warning in case the log thread did not stop correctly
      if cls.log_thread is not None and cls.log_thread.is_alive():
        cls.cls_log(logging.WARNING, "The Thread reading the log messages did "
                                     "not terminate in time !")

      # Checking whether all Blocks terminated gracefully
      if cls.instances and any(inst.is_alive() for inst in cls.instances):
        running = ', '.join(inst.name for inst in cls.instances
                            if inst.is_alive())
        cls.cls_log(logging.ERROR, f"Crappy failed to finish gracefully, "
                                   f"Block(s) {running} still running !")
        # An Exception is raised in case all the Blocks don't finish gracefully
        cls.raise_event.set()
        cls.cls_log(logging.WARNING, 'Set the raise Event because all the '
                                     'Blocks did not terminate as requested')
      else:
        cls.cls_log(logging.INFO, 'All Blocks done, Crappy terminated '
                                  'gracefully !\n')

    # Exceptions at that point cannot really be handled, but should still raise
    # in the main Process
    except (Exception, KeyboardInterrupt) as exc:
      # KeyboardInterrupt is a separate case
      if isinstance(exc, KeyboardInterrupt):
        cls.cls_log(logging.WARNING, "Caught KeyboardInterrupt while "
                                     "cleaning up, ignoring it !")
        # Special Event for the KeyboardInterrupt
        cls.kbi_event.set()
        cls.cls_log(logging.WARNING, 'Set the KbI Event after catching '
                                     'KeyboardInterrupt while cleaning up')
      else:
        cls.logger.exception("Caught exception while cleaning up !",
                             exc_info=exc)

        # Any Exception caught in the main Process must stop the script
        cls.raise_event.set()
        cls.cls_log(logging.WARNING, 'Set the raise Event after exception was '
                                     'caught in the main Process while '
                                     'cleaning up')

    # Deciding whether to raise and stop the main Process, and also resetting
    finally:
      # The try/finally is needed to reset Crappy before the exception is
      # raised but after the class Events are accessed
      try:
        # Deciding whether to raise or not
        if cls.raise_event.is_set() and not cls.no_raise:
          cls.cls_log(logging.ERROR, "An error occurred during Crappy's "
                                     "execution, raising CrappyFail !")
          raise CrappyFail
        elif cls.kbi_event.is_set() and not cls.no_raise:
          cls.cls_log(logging.ERROR, "KeyboardInterrupt called while running "
                                     "Crappy, raising it !")
          raise KeyboardInterrupt("Crappy was stopped using CTRL+C ! To "
                                  "disable this Exception, set the no_raise "
                                  "argument of crappy.start() or "
                                  "crappy.launch() to True.")
      finally:
        # Always resetting Crappy as the Blocks and synchronization objects
        # won't come in use anymore
        cls.reset()

  @classmethod
  def _set_logger(cls) -> None:
    """Initializes the logging for the main Process.

    It creates two Stream Loggers, one for the info and debug levels displaying
    on stdout and one for the other levels displaying on stderr. It also
    creates a File Logger for saving the log to a log file.

    The levels WARNING and above are always being displayed in the terminal, no
    matter what the user chooses. Similarly, the INFO log level and above are
    always being saved to the log file.
    """

    # The Logger handling all messages
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

      # Adding the handlers to the Logger
      crappy_log.addHandler(stream_handler)
      crappy_log.addHandler(stream_handler_err)
      if file_handler is not None:
        crappy_log.addHandler(file_handler)

    cls.logger = crappy_log

  @classmethod
  def stop_all(cls) -> None:
    """Method for stopping all the Blocks by setting the stop
    :obj:`~multiprocessing.Event`.
    
    .. versionremoved:: 2.0.0 *verbose* argument
    """

    if cls.stop_event is not None:
      cls.stop_event.set()
      cls.cls_log(logging.INFO, 'Stop event set after a call to stop(), all '
                                'Blocks should now finish')

  @classmethod
  def reset(cls) -> None:
    """Resets Crappy by emptying the :obj:`~weakref.WeakSet` containing
    references to all the Blocks and resetting the synchronization objects.

    This method is called at the very end of the
    :meth:`~crappy.blocks.Block._cleanup` method, but can also be called to
    "revert" the instantiation of Blocks while Crappy isn't started yet.
    """

    cls.instances = WeakSet()
    cls.names = list()
    cls.thread_stop = False
    cls.prepared_all = False
    cls.launched_all = False
    cls.no_raise = False

    cls.shared_t0 = None
    cls.ready_barrier = None
    cls.start_event = None
    cls.pause_event = None
    cls.stop_event = None
    cls.raise_event = None
    cls.kbi_event = None

    if cls.logger is not None:
      cls.cls_log(logging.INFO, 'Crappy was successfully reset')
  
  @classmethod
  def cls_log(cls, level: int, msg: str) -> None:
    """Wrapper for logging messages in the main Process.
    
    Ensures the Logger exists before trying to log, thus avoiding potential 
    errors.
    
    .. versionadded:: 2.0.0
    """
    
    if cls.logger is None:
      return
    cls.logger.log(level=level, msg=msg)

  @classmethod
  def _log_target(cls) -> None:
    """This method is the target to the Logger Thread.

    It reads log messages from a Queue and passes them to the Logger for
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
    """The method run by the Blocks when their :obj:`~multiprocessing.Process` 
    is started.

    It first calls :meth:`~crappy.blocks.Block.prepare`, then waits at the
    :obj:`~multiprocessing.Barrier` for all Blocks to be ready, then calls
    :meth:`~crappy.blocks.Block.begin`, then :meth:`~crappy.blocks.Block.main`,
    and finally :meth:`~crappy.blocks.Block.finish`.
    
    If an exception is raised, sets the shared stop 
    :obj:`~multiprocessing.Event` to warn all the other Blocks.
    """

    try:
      # Any Exception caught at the beginning should break the Barrier
      try:
        # Initializes the Logger for the Block
        self._set_block_logger()
        self.log(logging.INFO, "Block launched")

        # Running the preliminary actions before the test starts
        self.log(logging.INFO, "Block preparing")
        self.prepare()

      # If an Exception is raised, warning the other Blocks by breaking the
      # Barrier
      except (Exception, KeyboardInterrupt):
        self._ready_barrier.abort()
        self.log(logging.WARNING, "Broke the Barrier after an Exception was "
                                  "caught while preparing")
        raise

      # Waiting for all Blocks to be ready, except if the Barrier was broken
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
      self.log(logging.INFO, "Exiting main loop after stop Event was set")

    # A wrong data type was sent through a Link
    except LinkDataError:
      self.log(logging.ERROR, "Tried to send a wrong data type through a Link,"
                              " stopping !")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
    # An error occurred in another Block while preparing
    except PrepareError:
      self.log(logging.ERROR, "Exception raised in another Block while waiting"
                              " for all Blocks to be ready, stopping")
    # An error occurred in the CameraConfig window while preparing
    except CameraConfigError:
      self.log(logging.ERROR, "Exception raised in a configuration window, "
                              "stopping")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
    # An error occurred in a Camera process while preparing
    except CameraPrepareError:
      self.log(logging.ERROR, "Exception raised in a Camera Process while "
                              "preparing, stopping")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
      # An error occurred in a Camera Process while running
    except CameraRuntimeError:
      self.log(logging.ERROR, "Exception raised in a Camera process while "
                              "running, stopping")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
    # The start Event took too long to be set
    except StartTimeout:
      self.log(logging.ERROR, "Waited too long for start time to be set, "
                              "aborting !")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
    # Tried to access t0 but it's not set yet
    except T0NotSetError:
      self.log(logging.ERROR, "Trying to get the value of t0 when it's not "
                              "set yet, aborting")
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')
    # A Generator Block finished its Path
    except GeneratorStop:
      self.log(logging.WARNING, f"Generator Path exhausted, stopping the "
                                f"Block")
    # A FileReader Camera object has no more file to read from
    except ReaderStop:
      self.log(logging.WARNING, "Exhausted all the images to read from a "
                                "FileReader Camera, stopping the Block")
    # The user requested the script to stop
    except KeyboardInterrupt:
      self.log(logging.WARNING, f"KeyboardInterrupt caught, stopping")
      # A KeyboardInterrupt should stop the script and be raised as is
      self._kbi_event.set()
      self.log(logging.WARNING, 'Set the KbI Event after catching a '
                                'KeyboardInterrupt while running')
    # Another Exception occurred
    except (Exception,) as exc:
      self._logger.exception("Caught Exception while running !", exc_info=exc)
      # Any unexpected Exception should stop the script
      self._raise_event.set()
      self.log(logging.WARNING, 'Set the raise Event after catching an '
                                'unexpected Exception while running')

    # In all cases, trying to properly close the Block
    finally:
      try:
        self.log(logging.INFO, "Setting the stop Event")
        self._stop_event.set()
        self.log(logging.INFO, "Calling the finish method")
        self.finish()
      except KeyboardInterrupt:
        self.log(logging.WARNING, "Caught KeyboardInterrupt while finishing, "
                                  "ignoring it")
        # A KeyboardInterrupt should stop the script and be raised as is
        self._kbi_event.set()
        self.log(logging.WARNING, 'Set the KbI Event after catching a '
                                  'KeyboardInterrupt while finishing')
      except (Exception,) as exc:
        self._logger.exception("Caught Exception while finishing !",
                               exc_info=exc)
        # Any unexpected Exception should stop the script
        self._raise_event.set()
        self.log(logging.WARNING, 'Set the raise Event after catching an '
                                  'unexpected Exception while finishing')

  def main(self) -> None:
    """The main loop of the :meth:`~crappy.blocks.Block.run` method. Repeatedly
    calls the :meth:`~crappy.blocks.Block.loop` method and manages the looping
    frequency."""

    # Looping until told to stop or an error occurs
    while not self._stop_event.is_set():
      # Only looping if the Block is not paused
      if not self._pause_event.is_set() or not self.pausable:
        self.log(logging.DEBUG, "Looping")
        self.loop()
      else:
        self.log(logging.DEBUG, "Block currently paused, not calling loop()")
      # Handling the frequency in all cases to avoid hyperactive Blocks when
      # "paused"
      self.log(logging.DEBUG, "Handling freq")
      self._handle_freq()

  def prepare(self) -> None:
    """This method should perform any action required for initializing the
    Block before the test starts.

    For example, it can open a network connection, create a file, etc. It is
    also fine for this method not to be overridden if there's no particular
    action to perform.

    Note that this method is called once the :obj:`~multiprocessing.Process`
    associated to the Block has been started.
    """

    ...

  def begin(self) -> None:
    """This method can be considered as the first loop of the test, and is
    called before the :meth:`~crappy.blocks.Block.loop` method.

    It allows to perform initialization actions that cannot be achieved in the
    :meth:`~crappy.blocks.Block.prepare` method.
    """

    ...

  def loop(self) -> None:
    """This method is the core of the Block. It is called repeatedly during the
    test, until the test stops or an error occurs.

    Only in this method should data be sent to downstream Blocks, or received
    from upstream Blocks.

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

    Note that this method will normally be called even in case an error occurs,
    although that cannot be guaranteed.
    """

    ...

  def stop(self) -> None:
    """This method stops all the running Blocks.

    It should be called from the :meth:`~crappy.blocks.Block.loop` method of a
    Block. It allows to stop the execution of the script in a clean way,
    without raising an exception. It is mostly intended for users writing their
    own Blocks.

    Note:
      Calling this method in :meth:`~crappy.blocks.Block.__init__`,
      :meth:`~crappy.blocks.Block.prepare` or
      :meth:`~crappy.blocks.Block.begin` is not recommended, as the Block will
      only stop when reaching the :meth:`~crappy.blocks.Block.loop` method.
      Calling this method during :meth:`~crappy.blocks.Block.finish` will have
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
      while self._last_t + 1 / self.freq - t > 0:
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
    """Initializes the Logger for the Block.

    If the :mod:`multiprocessing` start method is `spawn` (mostly on Windows),
    redirects the log messages to a Queue for passing them to the main Process.
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

    If :obj:`False` (the default), only displays the :obj:`~logging.INFO`
    logging level. If :obj:`True`, displays the :obj:`~logging.DEBUG` logging
    level for the Block. And if :obj:`None`, displays only the
    :obj:`~logging.CRITICAL` logging level, which is equivalent to no
    information at all.
    
    .. versionadded:: 2.0.0
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
    shared between all the Blocks.
    
    .. versionadded:: 2.0.0
    """

    if self._instance_t0 is not None and self._instance_t0.value > 0:
      self.log(logging.DEBUG, "Start time value requested")
      return self._instance_t0.value
    else:
      raise T0NotSetError

  def add_output(self, link: Link) -> None:
    """Adds an output :class:`~crappy.links.Link` to the list of output Links
    of the Block."""

    self.outputs.append(link)

  def add_input(self, link: Link) -> None:
    """Adds an input :class:`~crappy.links.Link` to the list of input Links of
    the Block."""

    self.inputs.append(link)

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the Block. This option should be
    preferred to calling :func:`print`.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      return
    self._logger.log(log_level, msg)

  def send(self, data: Optional[Union[dict[str, Any], Iterable[Any]]]) -> None:
    """Method for sending data to downstream Blocks.

    The exact same :obj:`dict` is sent to every downstream Block.

    This method accepts the data to send either as a :obj:`dict` or as another
    type of iterable (like a :obj:`list` or a :obj:`tuple`). If data is
    provided as a dict, it is sent as is. The keys of the dict then correspond
    to the labels. Otherwise, the values given as an iterable are first
    converted to a dict using the ``self.labels`` attribute containing the
    labels to use.

    It is up to the user to match the order of the values in the iterable with
    the order of the labels in ``self.labels``. If the number of labels and the
    number of values to send do not match, no error is raised but some data
    might not get sent.
    """

    # Just in case, not handling non-existing data
    if data is None:
      return

    # Case when the data to send is not given as a dict
    if not isinstance(data, dict):
      # First, checking that labels are provided
      if self.labels is None or not self.labels:
        self.log(logging.ERROR, "Trying to send data as an iterable, but no "
                                "labels are specified ! Please add a "
                                "self.labels attribute.")
        raise LinkDataError

      # Trying to convert iterable data to dict using the given labels
      try:
        self.log(logging.DEBUG, f"Converting {data} to dict before sending")
        data = dict(zip(self.labels, data))
      except TypeError:
        self.log(logging.ERROR, f"Cannot convert data to send (of type "
                                f"{type(data)}) to dict ! Please ensure that "
                                f"the data is given as an iterable, as well as"
                                f" self.labels.")
        raise

    # Sending the data to the downstream Blocks
    for link in self.outputs:
      self.log(logging.DEBUG, f"Sending {data} to Link {link.name}")
      link.send(data)

  def data_available(self) -> bool:
    """Returns :obj:`True` if there's data available for reading in at least
    one of the input :class:`~crappy.links.Link`.
    
    .. versionchanged:: 2.0.0 renamed from *poll* to *data_available*
    """

    self.log(logging.DEBUG, "Data availability requested")
    return self.inputs and any(link.poll() for link in self.inputs)

  def recv_data(self) -> dict[str, Any]:
    """Reads the first available values from each incoming
    :class:`~crappy.links.Link` and returns them all in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Also, the returned values are the oldest available in the Links. See
    :meth:`~crappy.blocks.Block.recv_last_data` for getting the newest
    available values.

    Important:
      If data is received over a same label from different Links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a single value
      for each key (usually a :obj:`float` or a :obj:`str`).
    
    .. versionchanged:: 2.0.0 renamed from *recv_all* to *recv_data*
    """

    ret = dict()

    for link in self.inputs:
      ret |= link.recv()

    self.log(logging.DEBUG, f"Called recv_data, got {ret}")
    return ret

  def recv_last_data(self, fill_missing: bool = True) -> dict[str, Any]:
    """Reads all the available values from each incoming
    :class:`~crappy.links.Link`, and returns the newest ones in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Important:
      If data is received over a same label from different Links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.

    Args:
      fill_missing: If :obj:`True`, fills up the missing data for the known
        labels. This way, the last value received from all known labels is
        always returned. It can of course not fill up missing data for labels
        that haven't been received yet.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a single value
      for each key (usually a :obj:`float` or a :obj:`str`).

    .. versionremoved:: 1.5.10 *num* argument
    .. versionadded:: 1.5.10 *blocking* argument
    .. versionremoved:: 2.0.0 *blocking* argument
    .. versionchanged:: 2.0.0 renamed from *get_last* to *recv_last_data*
    """

    # Initializing the buffer storing the last received values
    if self._last_values is None:
      self._last_values = [dict() for _ in self.inputs]

    ret = dict()

    # Storing the received values in the return dict and in the buffer
    for link, buffer in zip(self.inputs, self._last_values):
      data = link.recv_last()
      ret |= data
      buffer |= data

    # If requested, filling up the missing values in the return dict
    if fill_missing:
      for buffer in self._last_values:
        ret |= buffer

    self.log(logging.DEBUG, f"Called recv_last_data, got {ret}")
    return ret

  def recv_all_data(self,
                    delay: Optional[float] = None,
                    poll_delay: float = 0.1) -> dict[str, list[Any]]:
    """Reads all the available values from each incoming
    :class:`~crappy.links.Link`, and returns them all in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Important:
      If data is received over a same label from different Links, part of it
      will be lost ! Always avoid using a same label twice in a Crappy script.
      See the :meth:`~crappy.blocks.Block.recv_all_data_raw` method for
      receiving data with no loss.

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
      one available in the Link, the last item is the newest available.

    .. versionremoved:: 1.5.10 *num* argument
    .. versionadded:: 1.5.10 *blocking* argument
    .. versionremoved:: 2.0.0 *blocking* argument
    .. versionchanged:: 2.0.0 renamed from *get_all_last* to *recv_all_data*
    """

    ret = defaultdict(list)
    t0 = time()

    # If simple recv_all, just receiving from all input links
    if delay is None:
      for link in self.inputs:
        ret |= link.recv_chunk()

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
                        poll_delay: float = 0.1) -> list[dict[str, list[Any]]]:
    """Reads all the available values from each incoming
    :class:`~crappy.links.Link`, and returns them separately in a list of
    dicts.

    Unlike :meth:`~crappy.blocks.Block.recv_all_data` this method does not fuse
    the received data into a single :obj:`dict`, so it is guaranteed to return
    all the available data with no loss.

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
    
    .. versionadded:: 2.0.0
    """

    ret = [defaultdict(list) for _ in self.inputs]
    t0 = time()

    # If simple recv_all, just receiving from all input links
    if delay is None:
      for dic, link in zip(ret, self.inputs):
        dic |= link.recv_chunk()

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

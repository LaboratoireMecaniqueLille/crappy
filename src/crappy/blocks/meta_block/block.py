# coding: utf-8

from __future__ import annotations
from platform import system
from multiprocessing import (Process, Value, Barrier, Event, Queue,
                             get_start_method, synchronize, queues,
                             sharedctypes, Manager, managers, Pipe, connection,
                             resource_tracker)
from threading import (BrokenBarrierError, Thread, Event as ThreadEvent,
                       current_thread)
from queue import Empty
import logging
import logging.handlers
from time import time, monotonic
from weakref import WeakSet
from types import TracebackType
from typing import Any, Protocol, runtime_checkable
from collections.abc import Sequence
from collections import defaultdict
import subprocess
from sys import stdout, stderr, argv
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import replace
from ctypes import c_byte

from ...links import Link, link_graph, GraphStructureError
from ..._exceptions import (LinkDataError, StartTimeout, StopRequested,
                            T0NotSetError, GeneratorStop, ReaderStop,
                            SchedulerStop, DefinitionError, ProcessExitError,
                            ShutdownTimeoutError)
from ..._failure import FailureManager


@runtime_checkable
class VisionBlockType(Protocol):
  """Protocol for the extra methods implemented by the
  :class:`~crappy.blocks.vision.block.VisionBlock`

  .. versionadded:: 2.2.0
  """

  def request_config(self, source: str) -> Any:
    """Method for requesting configuration information from an image source
    Block."""

  def add_config_request_in(self, request) -> None:
    """Registers a configuration request received from a downstream Block."""

  def add_config_request_out(self, request) -> None:
    """Registers a configuration request sent to an upstream image source."""

  def set_shared_objects(self) -> None:
    """Creates and distributes synchronization objects for output images."""


class Block(Process, ABC):
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

  The public execution settings
  :attr:`~crappy.blocks.meta_block.block.Block.niceness`,
  :attr:`~crappy.blocks.meta_block.block.Block.labels`,
  :attr:`~crappy.blocks.meta_block.block.Block.freq`,
  :attr:`~crappy.blocks.meta_block.block.Block.display_freq`,
  :attr:`~crappy.blocks.meta_block.block.Block.name`,
  :attr:`~crappy.blocks.meta_block.block.Block.pausable`, and
  :attr:`~crappy.blocks.meta_block.block.Block.is_vision_block` are validated
  properties. Child classes should set these public properties rather than
  their private backing attributes.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.8 remove metaclass and perform checks in
     __init_subclass__
  .. versionchanged:: 2.1.0 execution settings are exposed as validated
     properties rather than plain attributes
  .. versionchanged:: 2.1.0 removed management of the FT232H USB server
  """

  instances: WeakSet[Block] = WeakSet()
  _run_blocks: tuple[Block, ...] = tuple()
  names: list[str] = list()
  log_level: int | None = logging.DEBUG

  # The synchronization objects will be set later
  shared_t0: sharedctypes.Synchronized | None = None
  ready_barrier: synchronize.Barrier | None = None
  start_event: synchronize.Event | None = None
  pause_event: synchronize.Event | None = None
  stop_event: synchronize.Event | None = None
  logger: logging.Logger | None = None
  log_handlers: list[logging.Handler] = list()
  logger_state: tuple[int, bool, bool] | None = None
  log_queue: queues.Queue | None = None
  log_thread: Thread | None = None
  thread_stop: bool = False
  no_raise: bool = False

  # Objects used for managing Exception handling
  run_failure_manager: FailureManager | None = None
  failure_queues: tuple[queues.Queue, ...] = tuple()
  failure_thread: Thread | None = None
  failure_thread_stop: ThreadEvent | None = None
  watchdog_stop: ThreadEvent | None = None
  watchdog_thread: Thread | None = None
  early_exits: set[Block] = set()

  # Used for ImageLinks
  shared_mgr: managers.SyncManager | None = None

  prepared_all: bool = False
  launched_all: bool = False

  classes = dict()

  def __init_subclass__(cls, **kwargs) -> None:
    """Used for checking that two subclasses don't share the same name."""

    super().__init_subclass__()
    if cls.__name__ in cls.classes:
      raise DefinitionError(f"A Block with the name {cls.__name__} is already "
                            f"defined !")
    cls.classes[cls.__name__] = cls

  def __init__(self) -> None:
    """Sets the attributes and initializes the parent class."""

    Process.__init__(self)
    ABC.__init__(self)

    # The lists of input and output links
    self.outputs: list[Link] = list()
    self.inputs: list[Link] = list()

    # Hidden containers for properties, can be set by children classes
    self._display_freq: bool = False
    self._niceness: int = 0
    self._labels: Sequence[str] | None = None
    self._freq: float | None = None
    self._name: str = self.get_name(type(self).__name__)
    self._pausable: bool = True
    self._is_vision_block: bool = False

    # The synchronization objects will be set later
    self._instance_t0: sharedctypes.Synchronized | None = None
    self._ready_barrier: synchronize.Barrier | None = None
    self._start_event: synchronize.Event | None = None
    self._pause_event: synchronize.Event | None = None
    self._stop_event: synchronize.Event | None = None
    self._main_entered: c_byte | None = None

    # Configuration Pipe endpoints that this Process inherits under fork but
    # does not own. They are closed as soon as the child Process starts.
    self._config_connections_to_close: list[connection.Connection] = list()

    # Will be set later once the Blocks are starting
    self._failure_queue: queues.Queue | None = None
    self._failure_manager: FailureManager

    # The objects for logging will be set later
    self._log_queue: queues.Queue | None = None
    self._logger: logging.Logger | None = None
    self._debug: bool | None = False
    self._log_level: int | None = logging.INFO

    # Objects for displaying performance information about the block
    self._last_t: float = float('-inf')
    self._last_fps: float = float('-inf')
    self._n_loops: int = 0

    self._last_values: list[dict[str, Any]] = list()

    # Add node to the LinkGraph
    link_graph.add_node(self.name, type(self))

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
                log_level: int | None = logging.DEBUG,
                no_raise: bool = False) -> None:
    """Runs the Blocks and waits until the test and cleanup finish.

    Use :meth:`~crappy.blocks.meta_block.block.Block.prepare_all`,
    :meth:`~crappy.blocks.meta_block.block.Block.renice_all` and
    :meth:`~crappy.blocks.meta_block.block.Block.launch_all`
    separately when the script needs control between startup stages.

    Args:
      allow_root: If :obj:`True`, allow priority adjustment with sudo on
        Linux. The script must already have the required privileges.

        .. versionchanged:: 2.0.0 renamed from *high_prio* to *allow_root*
      log_level: Minimum severity of Crappy log messages, such as
        :obj:`logging.DEBUG` or :obj:`logging.WARNING`.
        :obj:`None` disables Crappy logging, not failure collection.

        .. versionadded:: 2.0.0
      no_raise: If :obj:`True`, suppress the final failure group only after
        every participating Block has entered
        :meth:`~crappy.blocks.meta_block.block.Block.main`. Failures in setup,
        preparation or :meth:`~crappy.blocks.meta_block.block.Block.begin`
        still raise. Once all Blocks have entered ``main()``, suppression
        includes interruptions and cleanup failures, but reporting and cleanup
        still take place. See :ref:`lifecycle-no-raise`.

        .. versionadded:: 2.0.0

    Raises:
      BaseExceptionGroup: Failures recorded during the run, unless
        ``no_raise`` applies. Ordinary failures form an :exc:`ExceptionGroup`,
        interruptions may require a :exc:`BaseExceptionGroup`.

    .. versionremoved:: 2.0.0 *t0*, *verbose*, *bg* arguments
    .. versionchanged:: 2.2.0
       Raises failures together after cleanup, including single failures.
    """

    # Make sure that the Exception Manager is running
    cls.no_raise = no_raise
    manager = cls._ensure_failure_manager()

    # Guard with the FailureManager to catch any Exception
    with manager.collect('start_all', raise_on_exit=False):
      cls._prepare_all(log_level)
      cls._renice_all(allow_root)
      cls._launch_all()
    cls._finalize_run()

  @classmethod
  def prepare_all(cls, log_level: int | None = logging.DEBUG) -> None:
    """Prepares all Blocks for a later call to
    :meth:`~crappy.blocks.meta_block.block.Block.launch_all`.

    Starts the Blocks and their
    :meth:`~crappy.blocks.meta_block.block.Block.prepare` methods. They wait
    for :meth:`~crappy.blocks.meta_block.block.Block.launch_all` before
    beginning the test. If this call fails, Crappy cleans up the partially
    prepared run before raising its failures.

    Args:
      log_level: Minimum severity of Crappy log messages.
        :obj:`None` disables Crappy logging, not failure collection.

        .. versionadded:: 2.0.0

    Raises:
      BaseExceptionGroup: Failures encountered while preparing or cleaning up.

    .. versionremoved:: 2.0.0 *verbose* argument
    .. versionchanged:: 2.2.0 raises recorded failures together after cleanup
    """

    # Make sure that the Exception Manager is running
    manager = cls._ensure_failure_manager()

    # Guard with the FailureManager to catch any Exception
    with manager.collect('prepare_all', raise_on_exit=False):
      cls._prepare_all(log_level)
      return
    # Only runs when an Exception is caught
    cls._finalize_run()

  @classmethod
  def renice_all(cls, allow_root: bool) -> None:
    """Adjusts Block process priorities on Linux and macOS.

    Does nothing on Windows. A refused priority change produces a warning.
    An exception while performing this operation stops and cleans up the run.

    Args:
      allow_root: If :obj:`True`, allow priority adjustment with sudo on
        Linux. The script must already have the required privileges.

        .. versionchanged:: 2.0.0 renamed from *high_prio* to *allow_root*

    Raises:
      BaseExceptionGroup: Failures encountered during this call or its cleanup.

    .. versionchanged:: 2.2.0
       Raises recorded failures together after cleanup.
    """

    # Make sure that the Exception Manager is running
    manager = cls._ensure_failure_manager()

    # Guard with the FailureManager to catch any Exception
    with manager.collect('renice_all', raise_on_exit=False):
      cls._renice_all(allow_root)
      return
    # Only runs when an Exception is caught
    cls._finalize_run()

  @classmethod
  def launch_all(cls, no_raise: bool = False) -> None:
    """Starts prepared Blocks and waits until the test and cleanup finish.

    Call after :meth:`~crappy.blocks.meta_block.block.Block.prepare_all`. All
    Blocks receive a common start time.

    Args:
      no_raise: If :obj:`True`, suppress the final failure group only after
        every participating Block has entered
        :meth:`~crappy.blocks.meta_block.block.Block.main`. Failures in setup,
        preparation or :meth:`~crappy.blocks.meta_block.block.Block.begin`
        still raise. Reporting and cleanup always take place. See
        :ref:`lifecycle-no-raise`.

    Raises:
      BaseExceptionGroup: Recorded failures, unless ``no_raise`` applies.

    .. versionremoved:: 2.0.0 *t0*, *verbose* and *bg* arguments
    .. versionchanged:: 2.2.0
       Raises failures together after cleanup. Suppression requires every
       participating Block to have entered ``main()``.
    """

    # Make sure that the Exception Manager is running
    cls.no_raise = no_raise
    manager = cls._ensure_failure_manager()

    # Guard with the FailureManager to catch any Exception
    with manager.collect('launch_all', raise_on_exit=False):
      cls._launch_all()
    cls._finalize_run()

  @classmethod
  def stop_all(cls) -> None:
    """Method for stopping all the Blocks by setting the stop
    :obj:`~multiprocessing.Event`.

    .. versionremoved:: 2.0.0 *verbose* argument
    """

    if cls.stop_event is not None:
      # Make sure that the Exception Manager is running
      manager = cls._ensure_failure_manager()

      # Guard with the FailureManager to catch any Exception
      with manager.collect('stop_all'):
        with manager.step('request stop'):
          cls.stop_event.set()
        with manager.step('stop diagnostic'):
          cls.cls_log(logging.INFO, 'Stop event set after a call to stop(), '
                                    'all Blocks should now finish')

  @classmethod
  def reset(cls) -> None:
    """Resets Crappy by emptying the :obj:`~weakref.WeakSet` containing
    references to all the Blocks, clearing the
    :class:`~crappy.links.LinkGraph`, and resetting the synchronization
    objects.

    This method is called at the very end of the
    :meth:`~crappy.blocks.meta_block.block.Block._cleanup` method, but can also
    be called to "revert" the instantiation of Blocks while Crappy isn't
    started yet.
    """

    # Make sure that the Exception Manager is running
    manager = cls._ensure_failure_manager()
    manager.phase = 'logging shutdown'

    with manager.collect('reset'):
      # First, check the status of the log Thread
      log_thread_stopped = False
      logger_failed = manager.logger_failed
      manager.logger_failed = True
      with manager.step('check logging Thread'):
        log_thread_stopped = (cls.log_thread is None or
                              not cls.log_thread.is_alive())
        if log_thread_stopped:
          manager.logger_failed = logger_failed

      # Request the Exception thread to stop
      with manager.step('stop failure receiver'):
        if cls.failure_thread_stop is not None:
          cls.failure_thread_stop.set()
      # Also make sure that the watchdog Thread stops
      with manager.step('stop watchdog'):
        if cls.watchdog_stop is not None:
          cls.watchdog_stop.set()

      # Discard the class attributes that ar no longer used
      cls.log_queue = None
      cls.log_thread = None
      cls.failure_thread = None
      cls.failure_thread_stop = None
      cls.failure_queues = tuple()
      cls.watchdog_thread = None
      cls.watchdog_stop = None
      cls.early_exits = set()
      cls.run_failure_manager = None
      # Also discard the corresponding instance attributes
      for instance in cls._run_blocks:
        with manager.step(f'release run state of {instance.name}'):
          instance._failure_queue = None
          instance._log_queue = None
          instance._main_entered = None
      cls.instances = WeakSet()
      cls._run_blocks = tuple()
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
      cls.shared_mgr = None

      # Display the final success message
      with manager.step('reset diagnostic'):
        if cls.logger is not None and not manager.logger_failed:
          cls.cls_log(logging.INFO, 'Crappy was successfully reset')
      # Reset the LinkGraph
      with manager.step('reset graph'):
        link_graph.reset()

      # Properly close the various loggers that were set up
      logger = cls.logger
      handlers = cls.log_handlers
      cls.log_handlers = list()
      for handler in handlers:
        with manager.step('detach handler'):
          if logger is not None:
            logger.removeHandler(handler)
        if log_thread_stopped:
          with manager.step('flush handler'):
            handler.flush()
          with manager.step('close handler'):
            handler.close()

      # Reset the logger state completely
      with manager.step('restore logger'):
        if logger is not None and cls.logger_state is not None:
          level, disabled, propagate = cls.logger_state
          logger.setLevel(level)
          logger.disabled = disabled
          logger.propagate = propagate
      cls.logger_state = None
      cls.logger = None

  @classmethod
  def cls_log(cls, level: int, msg: str) -> None:
    """Logs a class-level message through Crappy's main logger.

    This method is the class-level counterpart of :meth:`log` for code
    coordinating all Blocks. It silently returns before the main logger is
    configured.

    Args:
      level: Logging level for the message.
      msg: Message to record.

    .. versionadded:: 2.0.0
    """

    if cls.logger is None:
      return
    cls.logger.log(level=level, msg=msg)

  @classmethod
  def _ensure_failure_manager(cls) -> FailureManager:
    """Returns the failure manager for the current run.

    Returns:
      :class:`~crappy._failure.FailureManager` for failures from the main
      process and its Blocks.

    .. versionadded:: 2.2.0
    """

    if cls.run_failure_manager is None:
      cls.run_failure_manager = FailureManager('main', 'bootstrap')
      if cls.logger is not None:
        cls.run_failure_manager._logger = cls.logger
    return cls.run_failure_manager

  @classmethod
  def _prepare_all(cls, log_level: int | None) -> None:
    """Allocates the resources and starts the Blocks.

    .. versionadded:: 2.2.0
    """

    # Make sure that the Exception Manager is running
    manager = cls.run_failure_manager
    assert manager is not None
    manager.phase = 'bootstrap'

    # Freeze the set of Blocks to start
    if not cls.prepared_all and not cls.launched_all:
      cls._run_blocks = tuple(cls.instances)

    # Make sure that a logger is configured
    cls.log_level = log_level
    cls._set_logger()
    if cls.logger is not None:
      manager._logger = cls.logger

    # Used for closing stale connections after the Blocks have started
    parent_config_connections: list[connection.Connection] = list()
    # Associates each Block with the configuration Pipe endpoints it owns
    config_connections_by_owner: dict[
      str, list[connection.Connection]] = defaultdict(list)

    # Let the FailureManager catch Exceptions
    with manager.collect('preparation'):
      with manager.step('setup'):
        # Making sure that the Block classmethods are called in the right order
        if cls.prepared_all:
          raise RuntimeError("prepare_all was already called")

        if cls.launched_all:
          raise RuntimeError("Cannot prepare Blocks after launch_all")

        # Freeze the current Blocks before allocating resources
        blocks = cls._run_blocks
        instances_lookup = {instance.name: instance for instance in blocks}
        graph_names = set(link_graph.nodes)
        instance_names = set(instances_lookup)

        # Abort in case of inconsistency with the Blocks registered in graph
        missing = sorted(graph_names - instance_names)
        unregistered = sorted(instance_names - graph_names)
        if missing or unregistered:
          details = list()
          # Case when a Block is in the graph but not in the instances
          if missing:
            details.append(f"Blocks missing before startup (possibly "
                           f"garbage-collected): {', '.join(missing)}")
          # Case when a block is in the instances but not in the graph
          if unregistered:
            details.append(f"Blocks absent from the LinkGraph: "
                           f"{', '.join(unregistered)}")
          raise GraphStructureError('; '.join(details))
        cls._run_blocks = blocks

        # Displaying the first messages
        cls.cls_log(logging.INFO,
                    "===================== CRAPPY =====================")
        cls.cls_log(logging.INFO, f'Starting the script {argv[0]}\n')
        cls.cls_log(logging.INFO, 'Logger configured')

        # Setting all the synchronization objects at the class level
        cls.ready_barrier = Barrier(len(cls._run_blocks) + 1)
        cls.shared_t0 = Value('d', -1.0)
        cls.start_event = Event()
        cls.pause_event = Event()
        cls.stop_event = Event()
        cls.cls_log(logging.INFO, 'Multiprocessing synchronization objects '
                                  'set for main process')

        # Initializing the objects required for Blocks to log
        cls.log_queue = Queue()
        # Pass log queue explicitly so that the Thread keeps the same queue
        # even if a new one is created at the class level
        cls.log_thread = Thread(target=cls._log_target,
                                args=(cls.log_queue,),
                                name='crappy.log-reader', daemon=True)
        if get_start_method() != 'fork':
          cls.log_thread.start()
          cls.cls_log(logging.INFO, 'Logger thread started')

        # Passing the synchronization and logging objects to each Block
        for instance in cls._run_blocks:
          instance._failure_queue = Queue()
          cls.failure_queues += (instance._failure_queue,)
          instance._ready_barrier = cls.ready_barrier
          instance._instance_t0 = cls.shared_t0
          instance._stop_event = cls.stop_event
          instance._pause_event = cls.pause_event
          instance._start_event = cls.start_event
          instance._main_entered = Value('b', 0, lock=False)
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

        # Build the connection graph of the VisionBlocks and share
        # configuration requests to relevant sources
        if any(instance.is_vision_block for instance in cls._run_blocks):

          # Manage config requests for each source-consumer pair
          for source_name in link_graph.img_sources():
            source = instances_lookup[source_name]
            if not isinstance(source, VisionBlockType):
              raise TypeError(f"The image source {source_name} does not "
                              f"implement all the required methods")
            for consumer_name in link_graph.descendants(source_name,
                                                        kind='image'):
              consumer = instances_lookup[consumer_name]
              # If there's a configuration request from the consumer, share it
              # with the source
              if (isinstance(consumer, VisionBlockType) and
                  (config_request := consumer.request_config(source_name))
                  is not None):
                if config_request.requester != consumer.name:
                  raise ValueError("The name of the config requester doesn't "
                                   "match the name of the downstream Block")
                if config_request.img_source != source.name:
                  raise ValueError("The name of the image source doesn't "
                                   "match the name of the upstream Block")
                # Create a Pipe for sharing config data between Blocks
                recv_conn, send_conn = Pipe(duplex=False)
                parent_config_connections.extend((recv_conn, send_conn))
                config_connections_by_owner[source.name].append(send_conn)
                config_connections_by_owner[consumer.name].append(recv_conn)
                # Duplicate ConfigRequest and assign each an end of the Pipe
                source_request = replace(config_request, connection=send_conn)
                consumer_request = replace(config_request,
                                           connection=recv_conn)
                cls.cls_log(logging.DEBUG, f"Created config request: "
                                           f"{config_request}, now sharing it")
                # Register request in both the source and the consumer Blocks
                source.add_config_request_in(source_request)
                consumer.add_config_request_out(consumer_request)

          # With fork, every child inherits every open file descriptor from the
          # parent. Explicitly tell each Block which endpoints it must close on
          # startup so that a dead Block can still be detected via EOF
          if get_start_method() == 'fork':
            for instance in cls._run_blocks:
              owned_connections = {id(conn) for conn in
                                   config_connections_by_owner[instance.name]}
              instance._config_connections_to_close = [
                  conn for conn in parent_config_connections
                  if id(conn) not in owned_connections]

          cls.cls_log(logging.INFO, "Shared configuration requests from "
                                    "consumer vision Blocks to their sources")
        else:
          cls.cls_log(logging.INFO, "No vision Block detected, not building "
                                    "graph nor sharing config requests")

        # Initialize the common Manager for image Blocks if needed
        if any(instance.is_vision_block for instance in cls._run_blocks):
          cls.shared_mgr = Manager()
          cls.cls_log(logging.INFO, 'Created the shared Manager object')

          # Making vision Blocks generate their shared synchronization objects
          for instance in cls._run_blocks:
            if (instance.is_vision_block and
                isinstance(instance, VisionBlockType)):
              instance.set_shared_objects()
              cls.cls_log(logging.INFO, f"Set shared image-related objects "
                                        f"for the {instance.name} Block")
        else:
          cls.cls_log(logging.INFO, "No vision Block detected, not "
                                    "instantiating a shared Manager")

        # Under fork, ensures proper management of SharedMemory tracking at the
        # Process-level
        if (get_start_method() == 'fork' and
            any(instance.is_vision_block for instance in cls._run_blocks)):
          resource_tracker.ensure_running()
          cls.cls_log(logging.INFO, 'Shared-memory resource tracker started')

        # Starting all the Blocks
        for instance in cls._run_blocks:
          instance.start()
          cls.cls_log(logging.INFO, f'Started the {instance.name} Block')

        # Start the Thread that manages Exception communication between Blocks
        if cls.failure_queues:
          cls.failure_thread_stop = ThreadEvent()
          cls.failure_thread = Thread(target=cls._failure_target,
                                      name='crappy.failure-reader',
                                      daemon=True)
          cls.failure_thread.start()

        # Setting the prepared flag
        cls.prepared_all = True

    # Close the stale connections between VisionBlocks as this Process should
    # not unnecessarily own them
      for conn in parent_config_connections:
        with manager.step('close parent configuration endpoint'):
          conn.close()

  @classmethod
  def _renice_all(cls, allow_root: bool) -> None:
    """Runs the niceness commands for the prepared Blocks.

    .. versionadded:: 2.2.0
    """

    # Make sure that the Exception Manager is running
    manager = cls.run_failure_manager
    assert manager is not None
    manager.phase = 'renice'

    # Making sure that the Block classmethods are called in the right order
    if not cls.prepared_all or cls.launched_all:
      raise RuntimeError("renice_all requires a prepared, unlaunched run")

    # Renicing is not supported on Windows
    if system() == 'Windows':
      cls.cls_log(logging.INFO, "Not renicing processes on Windows")
      return

    # Renicing all the Blocks
    for instance in cls._run_blocks:
      # If root is not allowed then the minimum niceness is 0
      niceness = max(instance.niceness, -20 if allow_root else 0)
      command = ['renice', str(niceness), '-p', str(instance.pid)]
      if niceness < 0:
        command.insert(0, 'sudo')
      result = subprocess.call(command, stdout=subprocess.DEVNULL)
      if result:
        cls.cls_log(logging.WARNING,
                    f"Could not renice {instance.name} with "
                    f"{' '.join(command)} (return code {result})")
      else:
        cls.cls_log(logging.INFO,
                    f"Reniced process {instance.name} with PID {instance.pid} "
                    f"to niceness {niceness}")

  @classmethod
  def _launch_all(cls) -> None:
    """Synchronizes startup and uses the existing process-exit wait.

    .. versionadded:: 2.2.0
    """

    # Make sure that the Exception Manager is running
    manager = cls.run_failure_manager
    assert manager is not None
    manager.phase = 'synchronize'

    # Making sure that the Block classmethods are called in the right order
    if not cls.prepared_all or cls.launched_all:
      raise RuntimeError("launch_all requires a prepared, unlaunched run")
    cls.launched_all = True
    if cls.ready_barrier is None:
      raise RuntimeError("The ready Barrier isn't initialized")

    # Set up a watchdog Thread preventing deadlocks at the Barrier
    cls.watchdog_stop = ThreadEvent()
    cls.watchdog_thread = Thread(target=cls._watchdog_target,
                                 name='crappy.barrier-watchdog', daemon=True)

    # The Barrier waits for the main Process to be ready
    with manager.collect('readiness'):
      with manager.step('wait for Blocks'):
        cls.watchdog_thread.start()
        cls.cls_log(logging.INFO, 'Waiting for all Blocks to be ready')
        try:
          cls.ready_barrier.wait()
        # A failure happened in another Block
        except BrokenBarrierError:
          raise StopRequested from None
        # Handle race condition between ready and broken Barrier
        if cls.ready_barrier.broken:
          raise StopRequested

      # Once the Barrier is broken, the watchdog Thread should stop
      with manager.step('stop watchdog'):
        cls.watchdog_stop.set()
      # If the watchdog Thread doesn't stop, don't proceed any further
      with manager.step('join watchdog'):
        if cls.watchdog_thread.ident is not None:
          cls.watchdog_thread.join(timeout=0.5)
          if cls.watchdog_thread.is_alive():
            raise ShutdownTimeoutError("The Barrier watchdog did not stop",
                                       phase='synchronize', timeout=0.5)

    # Double-check for race conditions before proceeding
    if cls.ready_barrier.broken or (cls.stop_event is not None and
                                    cls.stop_event.is_set()):
      raise StopRequested

    # Initialize some shared synchronization objects
    manager.phase = 'start'
    cls.cls_log(logging.INFO, 'All Blocks ready now')
    if cls.shared_t0 is None or cls.start_event is None:
      raise RuntimeError("The start synchronization objects aren't "
                         "initialized")
    cls.shared_t0.value = time()
    cls.cls_log(logging.INFO, f'Start time set to {cls.shared_t0.value}s')
    cls.start_event.set()
    cls.cls_log(logging.INFO, 'Start event set, all Blocks can now start')

    # Until a Block finishes or crashes, nothing to do here
    manager.phase = 'main'
    cls.cls_log(logging.INFO, 'Main Process done, waiting for all Blocks to '
                              'finish')
    sentinels = [instance.sentinel for instance in cls._run_blocks]
    while sentinels and (cls.stop_event is None or
                         not cls.stop_event.is_set()):
      if connection.wait(sentinels, timeout=0.1):
        cls.cls_log(logging.INFO, "A Block has finished, waiting for the "
                                  "others")
        break

  @classmethod
  def _cleanup(cls) -> None:
    """Stops every Block, shared Manager, and logging resource independently.

    It first waits for all the Blocks to end, terminates survivors, and kills
    those still running after termination. Then, it stops the log_thread and
    records a failure if Processes or the logging Thread would still be
    running.

    Finally, it raises an exception if needed, in order to stop the script of
    the main Process. This way, any action that could follow the normal
    execution of Crappy won't happen, unless the user explicitly catches
    Crappy's exception and decides to go on with the script.
    """

    # Make sure that the Exception Manager is running
    manager = cls._ensure_failure_manager()
    manager.phase = 'framework cleanup'
    
    # Guard with the FailureManager to catch any Exception
    with manager.collect('Block cleanup', raise_on_exit=False):
      with manager.step('start log receiver'):
        if (get_start_method() != 'fork' and cls.log_thread is not None and
            cls.log_thread.ident is None):
          cls.log_thread.start()
      # Make sure that the Exception Thread is running
      with manager.step('start failure receiver'):
        if cls.failure_thread is None and cls.failure_queues:
          cls.failure_thread_stop = ThreadEvent()
          cls.failure_thread = Thread(target=cls._failure_target,
                                      name='crappy.failure-reader',
                                      daemon=True)
          cls.failure_thread.start()

      # Raise all the possible shutdown flags
      with manager.step('request shutdown'):
        if cls.stop_event is not None:
          cls.stop_event.set()
      with manager.step('abort readiness'):
        if (cls.ready_barrier is not None and
            (cls.start_event is None or not cls.start_event.is_set())):
          cls.ready_barrier.abort()
      with manager.step('stop watchdog'):
        if cls.watchdog_stop is not None:
          cls.watchdog_stop.set()

      # Waiting at most 10 seconds for all the Blocks to finish
      pending = dict()
      # First, check which Blocks are still alive
      for instance in cls._run_blocks:
        with manager.step(f'inspect {instance.name}'):
          if instance.is_alive():
            pending[instance.sentinel] = instance
      # Check stopped blocks each 0.5s at most for 10 seconds
      deadline = monotonic() + 10.0
      while pending and (remaining := deadline - monotonic()) > 0:
        exited = None
        with manager.step('wait for Block exits'):
          exited = connection.wait(pending, timeout=min(0.5, remaining))
        if exited is None:
          break
        # Avoid race condition between sentinel available and Process end
        for sentinel in exited:
          instance = pending.pop(sentinel)
          with manager.step(f'join {instance.name}'):
            instance.join(timeout=max(0.1, deadline - monotonic()))

      # Detect the surviving Blocks
      forced = set()
      survivors = list()
      for instance in cls._run_blocks:
        alive = True
        with manager.step(f'check {instance.name}'):
          alive = instance.is_alive()
        if alive:
          survivors.append(instance)

      # Terminate every Block that did not stop on its own
      if survivors:
        for instance in survivors:
          with manager.step(f'shutdown deadline for {instance.name}'):
            raise ShutdownTimeoutError(f"Block {instance.name} did not finish "
                                       f"within 10 seconds",
                                       process_name=instance.name,
                                       pid=instance.pid,
                                       phase='framework cleanup',
                                       timeout=10.0)
          # The two contexts below are executed since the Exception is caught
          with manager.step(f'termination diagnostic for {instance.name}'):
            cls.cls_log(logging.WARNING, f'Terminating Block {instance.name}')
          with manager.step(f'terminate {instance.name}'):
            instance.terminate()
            forced.add(instance)

        # Waiting at most half a second for the survivors to terminate
        deadline = monotonic() + 0.5
        for instance in survivors:
          with manager.step(f'join after terminate {instance.name}'):
            instance.join(timeout=max(0.0, deadline - monotonic()))

        # Detect the Blocks that survived termination
        living = list()
        for instance in survivors:
          alive = True
          with manager.step(f'check {instance.name}'):
            alive = instance.is_alive()
          if alive:
            living.append(instance)
        survivors = living

        # Kill the remaining survivors
        if survivors:
          for instance in survivors:
            with manager.step(f'killing diagnostic for {instance.name}'):
              cls.cls_log(logging.WARNING, f'Killing Block {instance.name}')
            with manager.step(f'kill {instance.name}'):
              instance.kill()
              forced.add(instance)

          # Waiting at most half a second for the survivors to get killed
          deadline = monotonic() + 0.5
          for instance in survivors:
            with manager.step(f'join after kill {instance.name}'):
              instance.join(timeout=max(0.0, deadline - monotonic()))

          # Detect the Blocks that survived killing
          living = list()
          for instance in survivors:
            alive = True
            with manager.step(f'check {instance.name}'):
              alive = instance.is_alive()
            if alive:
              living.append(instance)

          # Log the running status of the Blocks
          status = ('Not all Blocks could be stopped even after killing them'
                    if living else 'All Blocks stopped after killing')
        else:
          status = 'All Blocks stopped after termination'
      else:
        status = 'All Blocks stopped'
      with manager.step('Block shutdown diagnostic'):
        cls.cls_log(logging.INFO, status)

      # Links own the parent-process copies of the communication endpoints
      links: set[Link] = set()
      for instance in cls._run_blocks:
        # First gather all the existing Links
        with manager.step(f'collect Links of {instance.name}'):
          links.update((*instance.inputs, *instance.outputs))
      # Then close all the gathered Links
      for link in links:
        with manager.step(f'Link closure diagnostic {link.name!r}'):
          cls.cls_log(logging.DEBUG, f'Closing Link {link.name!r}')
        with manager.step(f'close Link {link.name!r}'):
          link.close(manager=manager)

      # Stopping the shared Manager if required
      if cls.shared_mgr is not None:
        with manager.step('shared Manager shutdown diagnostic'):
          cls.cls_log(logging.INFO, "Stopping the shared Manager")
        with manager.step('shut down shared Manager'):
          cls.shared_mgr.shutdown()

      # Stop the Exception Thread
      with manager.step('request failure receiver stop'):
        # First, ask the Exception Thread to stop
        if cls.failure_thread_stop is not None:
          cls.failure_thread_stop.set()
      # Then, wait for it to finish
      with manager.step('join failure receiver'):
        if (cls.failure_thread is not None and
            cls.failure_thread.ident is not None):
          cls.failure_thread.join(timeout=0.5)
          # If it is still alive, report it as an error
          if cls.failure_thread.is_alive():
            raise ShutdownTimeoutError("The failure receiver did not stop",
                                       phase='monitoring', timeout=0.5)

      # Close the Exception Queues
      for queue in cls.failure_queues:
        with manager.step('close failure Queue'):
          queue.close()

      # Check proper Block termination and report unexpected failures
      reported = {report.block_name for report, _ in manager.snapshot()}
      for instance in cls._run_blocks:
        # Check exit codes of Blocks and report abnormal ones
        with manager.step(f'verify exit of {instance.name}'):
          code = instance.exitcode
          if (instance not in forced and
              instance.name not in reported and
              (instance in cls.early_exits or
               (isinstance(code, int) and code != 0))):
            phase = 'prepare' if instance in cls.early_exits else 'main'
            raise ProcessExitError(f"Block {instance.name} exited "
                                   f"unexpectedly (exit code {code})",
                                   process_name=instance.name,
                                   pid=instance.pid,
                                   phase=phase,
                                   exitcode=code)
        # Check run status of Blocks and report abnormal ones
        with manager.step(f'verify shutdown of {instance.name}'):
          if instance.is_alive():
            raise ShutdownTimeoutError(
                f"Block {instance.name} is still running after shutdown",
                process_name=instance.name, pid=instance.pid,
                phase='framework cleanup', timeout=11.0)

      # Stop the log Thread and wait for it to finish
      cls.thread_stop = True
      with manager.step('logging shutdown diagnostic'):
        if cls.log_thread is not None and not manager.logger_failed:
          cls.cls_log(logging.INFO, "Stopping the logging Thread")
      with manager.step('join logging Thread'):
        if cls.log_thread is not None and cls.log_thread.ident is not None:
          manager.logger_failed = True
          cls.log_thread.join(timeout=1.0)
          if cls.log_thread.is_alive():
            raise ShutdownTimeoutError("The logging Thread did not stop",
                                       phase='logging shutdown', timeout=1.0)
          manager.logger_failed = False

      # Close the log queue
      with manager.step('log Queue closure diagnostic'):
        if cls.log_queue is not None and not manager.logger_failed:
          cls.cls_log(logging.DEBUG, "Closing the log Queue")
      with manager.step('close log Queue'):
        if cls.log_queue is not None:
          cls.log_queue.close()
      with manager.step('shutdown diagnostic'):
        if not manager.logger_failed and not manager.snapshot():
          cls.cls_log(logging.INFO,
                      'All Blocks done, Crappy terminated gracefully !\n')

  @classmethod
  def _finalize_run(cls) -> None:
    """Cleans up, resets independently, then raises the run failures.

    .. versionadded:: 2.2.0
    """

    # Make sure that the Exception Manager is running
    manager = cls._ensure_failure_manager()
    no_raise = cls.no_raise
    eligible = False

    # Cleanup the Block attributes and reset
    with manager.collect('run finalization', raise_on_exit=False):
      with manager.step('cleanup'):
        cls._cleanup()
      with manager.step('main-entry eligibility'):
        eligible = bool(cls._run_blocks) and all(
            instance._main_entered is not None and instance._main_entered.value
            for instance in cls._run_blocks)
      with manager.step('reset'):
        cls.reset()

    # Report failures if all blocks entered the main phase
    if not no_raise or not eligible:
      manager.raise_if_failed('Crappy run failures', group_by_source=True)

  @classmethod
  def _set_logger(cls) -> None:
    """Initializes the logging for the main Process.

    It creates two Stream Loggers, one for the info and debug levels displaying
    on stdout and one for the other levels displaying on stderr. It also
    creates a File Logger for saving the log to a log file.

    All handlers respect the requested level. Disabling Crappy logging does
    not change the logging configuration of the rest of the application.
    """

    # Keep the existing logger if already defined
    if cls.logger is not None:
      return

    cls.log_handlers = list()
    # The Logger handling all messages
    crappy_log = logging.getLogger('crappy')
    cls.logger_state = (crappy_log.level, crappy_log.disabled,
                        crappy_log.propagate)
    cls.logger = crappy_log
    crappy_log.disabled = cls.log_level is None
    crappy_log.propagate = False

    if cls.log_level is not None:
      crappy_log.setLevel(cls.log_level)

    # In case there's no logging, no need to configure the handlers
    if cls.log_level is not None:
      # The two handlers for displaying messages in the console
      stream_handler = logging.StreamHandler(stream=stdout)
      stream_handler_err = logging.StreamHandler(stream=stderr)
      cls.log_handlers.extend((stream_handler, stream_handler_err))

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
        cls.log_handlers.append(file_handler)
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

  @classmethod
  def _log_target(cls, log_queue: queues.Queue) -> None:
    """This method is the target to the Logger Thread.

    It reads log messages from a Queue and passes them to the Logger for
    handling.
    """

    # Keep first instances in case the Block is reset
    manager = cls.run_failure_manager
    handlers = tuple(cls.log_handlers)
    stop_event = cls.stop_event
    ready_barrier = cls.ready_barrier
    # Means that Crappy was likely reset without this Thread stopping
    if cls.log_queue is not log_queue:
      return
    assert manager is not None

    # Main loop, simply receive log messages and handle them
    try:
      while cls.log_queue is log_queue:
        try:
          record = log_queue.get(block=True, timeout=0.05)
        except Empty:
          if cls.thread_stop:
            return
          continue

        # Drain the current run after stop, but never forward into a new run.
        if cls.log_queue is not log_queue:
          return
        logger = logging.getLogger(record.name)
        logger.handle(record)

    # Exceptions caught in the logger Thread are reported the regular way
    except BaseException as error:
      try:
        manager.capture_exception(error, phase='logging')
      except (BaseException,):
        pass

      # Request shutdown as soon as the Exception is reported.
      with manager.collect('log reader shutdown', raise_on_exit=False):
        with manager.step('request stop'):
          if stop_event is not None:
            stop_event.set()
        with manager.step('abort readiness'):
          if ready_barrier is not None:
            ready_barrier.abort()

    # In case reset() left logging in a stale state, close log handlers
    finally:
      if cls.log_queue is not log_queue and handlers:
        with manager.collect('abandoned log reader shutdown',
                             raise_on_exit=False):
          for handler in handlers:
            with manager.step('flush handler'):
              handler.flush()
            with manager.step('close handler'):
              handler.close()

  @classmethod
  def _failure_target(cls) -> None:
    """Receives Block failures and requests coordinated shutdown.

    .. versionadded:: 2.2.0
    """

    # Keep reference to the original objects in case reset() is called
    manager = cls.run_failure_manager
    incoming = cls.failure_queues
    thread_stop = cls.failure_thread_stop
    stop_event = cls.stop_event
    ready_barrier = cls.ready_barrier
    if (cls.run_failure_manager is None or thread_stop is None or
        cls.failure_thread is not current_thread()):
      return

    assert manager is not None

    stopping = False
    try:
      while True:
        # Receive incoming failure reports from the FailureManager
        received = manager.receive(incoming)
        # As soon as at least one Exception is received, means we have to stop
        if manager.snapshot():
          with manager.collect('failure notification'):
            if not stopping:
              stopping = True
              with manager.step('request stop'):
                if stop_event is not None:
                  stop_event.set()
              with manager.step('abort readiness'):
                if ready_barrier is not None:
                  ready_barrier.abort()

        # Only stop when there are no further Exceptions to collect
        if thread_stop.is_set():
          if not received:
            break
        # Sleep to lower CPU load
        elif not received:
          thread_stop.wait(0.05)

    # If an Exception is encountered, report it and make Blocks stop
    except BaseException as error:
      try:
        manager.capture_exception(error, phase='monitoring')
      except (BaseException,):
        pass
      with manager.collect('failure receiver shutdown', raise_on_exit=False):
        with manager.step('request stop'):
          if stop_event is not None:
            stop_event.set()
        with manager.step('abort readiness'):
          if ready_barrier is not None:
            ready_barrier.abort()

  @classmethod
  def _watchdog_target(cls) -> None:
    """Aborts startup if a Block exits before the watchdog is disarmed.

    .. versionadded:: 2.2.0
    """

    # Keep references to these objects even if the Block is reset
    manager = cls.run_failure_manager
    stop = cls.watchdog_stop
    barrier = cls.ready_barrier
    blocks = cls._run_blocks
    early_exits = cls.early_exits
    stop_event = cls.stop_event
    if (manager is None or stop is None or barrier is None or
        cls.watchdog_thread is not current_thread()):
      return

    # Guard with the FailureManager to catch any Exception
    with manager.collect('startup watchdog', raise_on_exit=False):
      with manager.step('watch readiness'):
        sentinels = {block.sentinel: block for block in blocks}
        while sentinels and not stop.is_set():
          exited = connection.wait(sentinels, timeout=0.1)
          # If a failure is detected by other means, our job is done
          if stop.is_set() or barrier.broken:
            return
          # Report any Block that unexpectedly stopped
          if exited:
            early_exits.update(sentinels[sentinel] for sentinel in exited)
            break
        # If a failure is detected by other means, our job is done
        else:
          return

      # If we're the first to witness the Exception, raise the failure flags
      with manager.step('request stop'):
        if stop_event is not None:
          stop_event.set()
      with manager.step('abort readiness'):
        barrier.abort()

  @staticmethod
  def _stdout_filter(rec: logging.LogRecord) -> bool:
    """Returns :obj:`True` if the input log message has level INFO or DEBUG,
    :obj:`False` otherwise."""

    return rec.levelno in (logging.DEBUG, logging.INFO)

  def run(self) -> None:
    """The method run by the Blocks when their :obj:`~multiprocessing.Process`
    is started.

    It first calls :meth:`~crappy.blocks.meta_block.block.Block.prepare`, then
    waits at the :obj:`~multiprocessing.Barrier` for all Blocks to be ready,
    then calls :meth:`~crappy.blocks.meta_block.block.Block.begin`, then
    :meth:`~crappy.blocks.meta_block.block.Block.main`, and finally
    :meth:`~crappy.blocks.meta_block.block.Block.finish`.

    If an exception is raised, sets the shared stop
    :obj:`~multiprocessing.Event` to warn all the other Blocks.
    """

    self._failure_manager = FailureManager(self.name, 'bootstrap')
    manager = self._failure_manager
    with manager.collect('Block lifecycle', raise_on_exit=False):
      with manager.step('execution'):
        # Start with setting the Exception queue and the logger
        manager.set_queue(self._failure_queue)
        self._set_block_logger()
        if self._logger is not None:
          manager._logger = self._logger
        self.log(logging.INFO, "Block launched")

        # Close inherited endpoints before running preparation work
        self._close_config_connections()

        # Prepare the Block
        manager.phase = 'prepare'
        self.log(logging.INFO, "Block preparing")
        self.prepare()

        # Wait at the Barrier for all Blocks to be ready
        manager.phase = 'synchronize'
        if self._ready_barrier is None:
          raise RuntimeError("The ready Barrier isn't initialized")
        if self._stop_event is None:
          raise RuntimeError("The stop Event isn't initialized")
        if self._stop_event.is_set() or manager.snapshot():
          raise StopRequested
        self.log(logging.INFO, "Waiting for the other Blocks to be ready")
        try:
          self._ready_barrier.wait()
        except BrokenBarrierError:
          self.log(logging.WARNING,
                   "Readiness cancelled by a broken Barrier")
          raise StopRequested from None
        self.log(logging.INFO, "All Blocks ready now")

        # Wait for the start Event to be set, for synchronization
        manager.phase = 'start'
        if self._start_event is None:
          raise RuntimeError("The start Event isn't initialized")
        if self._stop_event.is_set():
          raise StopRequested
        self.log(logging.INFO, "Waiting for the start time to be set")
        started = self._start_event.wait(timeout=1.0)
        if self._stop_event.is_set():
          raise StopRequested
        if not started:
          raise StartTimeout
        self.log(logging.INFO, "Start time set, Block starting")

        # Let the Block begin
        manager.phase = 'begin'
        self.log(logging.INFO, "Calling begin method")
        self.begin()

        # Run the main loops
        manager.phase = 'main'
        self._last_t = monotonic()
        self._last_fps = self._last_t
        self._n_loops = 0
        self.log(logging.INFO, "Entering main loop")
        if self._main_entered is None:
          raise RuntimeError("The main-entered flag isn't initialized")
        with manager.step('main',
                          completion=(GeneratorStop, SchedulerStop,
                                      ReaderStop)):
          self._main_entered.value = 1
          self.main()
          self.log(logging.INFO,
                   "Exiting main loop after stop Event was set")

      execution_phase = manager.phase
      manager.phase = 'framework cleanup'
      with manager.step('stop diagnostic'):
        self.log(logging.INFO, "Setting the stop Event")
      with manager.step('request stop'):
        if self._stop_event is not None:
          self._stop_event.set()
      with manager.step('abort preparation'):
        if (execution_phase in ('bootstrap', 'prepare', 'synchronize') and
            self._ready_barrier is not None):
          self._ready_barrier.abort()

      manager.phase = 'finish'
      with manager.step('finish diagnostic'):
        self.log(logging.INFO, "Calling the finish method")
      with manager.step('finish'):
        self.finish()

      manager.phase = 'framework cleanup'
      with manager.step('close inherited config Pipes'):
        self._close_config_connections()
      with manager.step('completion diagnostic'):
        self.log(logging.INFO, f"Block {self.name} done")

      # Disconnect reporting before closing and flushing its public Queue.
      manager.phase = 'failure Queue shutdown'
      if self._failure_queue is not None:
        queue = self._failure_queue
        self._failure_queue = None
        with manager.step('disconnect'):
          manager.set_queue(None)
        with manager.step('close'):
          queue.close()
        with manager.step('flush'):
          queue.join_thread()

  def _record_callback_failure(self, error: BaseException) -> None:
    """Reports a callback failure and requests the test to stop.

    Args:
      error: Exception or :exc:`BaseExceptionGroup` caught by the callback.

    .. versionadded:: 2.2.0
    """

    manager = self._failure_manager

    # Record the Exception and stop the test
    with manager.collect('Block callback', raise_on_exit=False):
      manager.record_context_exception(error, 'callback')
      with manager.step('request callback shutdown'):
        if self._stop_event is not None:
          self._stop_event.set()
      with manager.step('abort callback preparation'):
        if (self._ready_barrier is not None and
            (self._start_event is None or not self._start_event.is_set())):
          self._ready_barrier.abort()

  def _report_tk_callback_exception(self,
                                    _: type[BaseException],
                                    error: BaseException,
                                    traceback: TracebackType | None) -> None:
    """Reports a failed Tk callback and requests the test to stop.

    Args:
      _: Exception class supplied by Tk.
      error: Exception raised by the callback.
      traceback: Call stack supplied by Tk, or :obj:`None` if unavailable.

    .. versionadded:: 2.2.0
    """

    self._record_callback_failure(error.with_traceback(traceback))

  def _close_config_connections(self) -> None:
    """Closes inherited configuration endpoints, retaining failed ones.

    .. versionadded:: 2.2.0
    """

    if not self._config_connections_to_close:
      return

    with self._failure_manager.collect('configuration Pipe shutdown'):
      for conn in tuple(self._config_connections_to_close):
        with self._failure_manager.step('close inherited config Pipe'):
          conn.close()
          self._config_connections_to_close.remove(conn)

  def main(self) -> None:
    """The main loop of the :meth:`~crappy.blocks.meta_block.block.Block.run`
    method. Repeatedly calls the
    :meth:`~crappy.blocks.meta_block.block.Block.loop` method and manages the
    looping frequency."""

    if self._stop_event is None:
      raise RuntimeError("The stop Event doesn't exist, it should")
    # Looping until told to stop or an error occurs
    while not self._stop_event.is_set():
      if self._pause_event is None:
        raise RuntimeError("The pause Event doesn't exist, it should")
      # Record the last time the Block started a loop or had the opportunity to
      self._last_t = monotonic()
      # Only looping if the Block is not paused
      if not self._pause_event.is_set() or not self.pausable:
        self.log(logging.DEBUG, "Looping")
        self.loop()
      else:
        self.log(logging.DEBUG, "Block currently paused, not calling loop()")
      # Add the loop that just finished to the loop counter
      self._n_loops += 1
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
    called before the :meth:`~crappy.blocks.meta_block.block.Block.loop`
    method.

    It allows to perform initialization actions that cannot be achieved in the
    :meth:`~crappy.blocks.meta_block.block.Block.prepare` method.
    """

    ...

  @abstractmethod
  def loop(self) -> None:
    """This method is the core of the Block. It is called repeatedly during the
    test, until the test stops or an error occurs.

    Only in this method should data be sent to downstream Blocks, or received
    from upstream Blocks.

    Although it is possible not to override this method, that has no practical
    interest and this method should always be rewritten.
    """

    ...

  def finish(self) -> None:
    """Closes the resources used by this Block when the test ends.

    Override this method to close files, devices or connections. Allow for
    :meth:`~crappy.blocks.meta_block.block.Block.prepare` having failed before
    setup was complete. This method is normally called after an error too, but
    cannot run if the process is forcibly stopped.

    For several independent cleanup operations, use
    :meth:`~crappy._failure.FailureManager.collect` and
    :meth:`~crappy._failure.FailureManager.step` on ``self._failure_manager``
    so one failed close does not skip the others.

    .. versionchanged:: 2.2.0
       Cleanup failures are kept alongside failures from earlier stages.
    """

    ...

  def stop(self) -> None:
    """This method stops all the running Blocks.

    It should be called from the
    :meth:`~crappy.blocks.meta_block.block.Block.loop` method of a Block. It
    allows to stop the execution of the script in a clean way, without raising
    an exception. It is mostly intended for users writing their own Blocks.

    Note:
      Calling this method in
      :meth:`~crappy.blocks.meta_block.block.Block.__init__`,
      :meth:`~crappy.blocks.meta_block.block.Block.prepare` or
      :meth:`~crappy.blocks.meta_block.block.Block.begin` is not recommended,
      as the Block will only stop when reaching the
      :meth:`~crappy.blocks.meta_block.block.Block.loop` method. Calling this
      method during :meth:`~crappy.blocks.meta_block.block.Block.finish` will
      have no effect.
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

    # Only handling frequency if requested
    if self.freq is not None:
      next_t = self._last_t + 1 / self.freq
      assert self._stop_event is not None
      if self._stop_event.wait(max(0., next_t - monotonic())):
        return

    # Displaying frequency every 2 seconds
    if self.display_freq and self._last_t - self._last_fps > 2:
      self.log(logging.INFO,
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
    logger.disabled = self._log_level is None
    if self._log_level is not None:
      logger.setLevel(self._log_level)

    # On spawn and forkserver, the messages need to be sent through a Queue for
    # logging
    if get_start_method() != 'fork' and self._log_level is not None:

      if self._log_queue is None:
        raise RuntimeError("The log queue is required but was never set")

      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger

  @property
  def debug(self) -> bool | None:
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
  def debug(self, val: bool | None) -> None:
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

  @property
  def niceness(self) -> int:
    """Niceness of the Block on Unix systems, from `-20` to `19`.

    Not used on other systems (typically Windows).

    .. versionadded:: 2.1.0
    """

    return self._niceness

  @niceness.setter
  def niceness(self, val: int) -> None:
    if not isinstance(val, int):
      raise TypeError("The niceness must be an integer")
    if not -20 <= val <= 19:
      raise ValueError("The niceness must be between -20 and 19")
    self._niceness = val

  @property
  def labels(self) -> Sequence[str] | None:
    """Unique labels naming the values sent to downstream Blocks.

    Must be a sequence containing only strings, or :obj:`None` when data will
    be sent as dictionaries.

    .. versionadded:: 2.1.0
    """

    return self._labels

  @labels.setter
  def labels(self, val: Sequence[str] | None) -> None:
    # checking that the labels is not a string
    if val is not None and isinstance(val, str):
      raise TypeError("The labels must be a Sequence of strings or None")
    # Checking that the labels is a valid Sequence type
    if val is not None and not isinstance(val, Sequence):
      raise TypeError("The labels must be a Sequence of strings or None")
    # Checking that the labels is a Sequence of strings only
    if val is not None and not all(isinstance(label, str) for label in val):
      non_str = [label for label in val if not isinstance(label, str)]
      raise ValueError(f"Some labels are not strings: "
                       f"{', '.join(map(repr, non_str))}")
    # Checking that there are no duplicate labels
    if val is not None and len(set(val)) != len(val):
      raise ValueError("Duplicate labels provided in the list of labels!")

    self._labels = val

  @property
  def freq(self) -> float | None:
    """Target looping frequency of the Block, in Hz.

    Positive integer and floats are accepted. If :obj:`None`, the Block tries
    to loop as fast as possible.

    .. versionadded:: 2.1.0
    """

    return self._freq

  @freq.setter
  def freq(self, val: float | None) -> None:
    if (val is not None and not isinstance(val, float)
        and not isinstance(val, int)):
      raise TypeError("The freq must be a float or None")
    if val is not None and val <= 0:
      raise ValueError("The freq must be a strictly positive float")
    self._freq = None if val is None else float(val)

  @property
  def display_freq(self) -> bool:
    """Whether to periodically display the achieved frequency in the logs.

    .. versionadded:: 2.1.0
    """

    return self._display_freq

  @display_freq.setter
  def display_freq(self, val: bool) -> None:
    if not isinstance(val, bool):
      raise TypeError("display_freq must be a boolean")
    self._display_freq = val

  @property
  def name(self) -> str:
    """Unique, non-empty name used in logs and the Link connection graph.

    Assigning a new name before the Block starts also updates Crappy's global
    Block-name registry as well as the connection graph. A running Block cannot
    be renamed.

    .. versionadded:: 2.1.0
    """

    return self._name

  @name.setter
  def name(self, val: str) -> None:
    # Checking that the name is a valid string
    if not isinstance(val, str):
      raise TypeError("The name must be a string")
    if not val:
      raise ValueError("The name must be a non-empty string")
    # Checking that the name doesn't already exist
    if val != self._name and val in self.names:
      raise ValueError(f"The name {val} is already in use by another Block!")
    if self.is_alive():
      raise RuntimeError("Cannot edit a Block's name at runtime!")
    # Update the LinkGraph
    link_graph.rename_node(self._name, val)
    # Set new name and update the list of names currently in use
    if self._name in self.names:
      self.names.remove(self._name)
    self._name = val
    self.names.append(val)

  @property
  def pausable(self) -> bool:
    """Whether the Block can be paused by Crappy.

    A non-pausable Block keeps running while the other Blocks are paused.

    .. versionadded:: 2.1.0
    """

    return self._pausable

  @pausable.setter
  def pausable(self, val: bool) -> None:
    if not isinstance(val, bool):
      raise TypeError("pausable must be a boolean")
    self._pausable = val

  @property
  def is_vision_block(self) -> bool:
    """Whether the Block can be connected through a
    :class:`~crappy.links.img_link.ImageLink`.

    This property is normally set by image-oriented Block base classes rather
    than by users.

    .. versionadded:: 2.1.0
    """

    return self._is_vision_block

  @is_vision_block.setter
  def is_vision_block(self, val: bool) -> None:
    if not isinstance(val, bool):
      raise TypeError("is_vision_block must be a boolean")
    if not isinstance(self, VisionBlockType):
      raise TypeError(f"The Block {self.name} does not implement all the "
                      f"required methods of a VisionBlock!")
    self._is_vision_block = val

  def add_output(self, link: Link) -> None:
    """Adds an output :class:`~crappy.links.link.Link` to the list of output
    Links of the Block."""

    self.outputs.append(link)

  def add_input(self, link: Link) -> None:
    """Adds an input :class:`~crappy.links.link.Link` to the list of input
    Links of the Block."""

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

  def send(self, data: dict[str, Any] | Sequence[Any] | None) -> None:
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
    might not get sent. If ``data`` is :obj:`None` or resolves to an empty
    :obj:`dict`, nothing is sent.
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

    # Not sending an empty data dictionary
    if not data:
      self.log(logging.WARNING, "The data dictionary to send is empty, not "
                                "sending")
      return

    # Sending the data to the downstream Blocks
    for link in self.outputs:
      self.log(logging.DEBUG, f"Sending data for labels "
                              f"{', '.join(data.keys())} to Link {link.name}")
      link.send(data)

  def data_available(self) -> bool:
    """Returns :obj:`True` if there's data available for reading in at least
    one of the input :class:`~crappy.links.link.Link`.
    
    .. versionchanged:: 2.0.0 renamed from *poll* to *data_available*
    """

    self.log(logging.DEBUG, "Data availability requested")
    return bool(self.inputs) and any(link.poll() for link in self.inputs)

  def recv_data(self) -> dict[str, Any]:
    """Reads the first available values from each incoming
    :class:`~crappy.links.link.Link` and returns them all in a single dict.

    The returned :obj:`dict` might not always have a fixed number of keys,
    depending on the availability of incoming data.

    Also, the returned values are the oldest available in the Links. See
    :meth:`~crappy.blocks.meta_block.block.Block.recv_last_data` for getting
    the newest available values.

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

    self.log(logging.DEBUG, f"Called recv_data, got data for labels "
                            f"{', '.join(ret.keys())}")
    return ret

  def recv_last_data(self, fill_missing: bool = True) -> dict[str, Any]:
    """Reads all the available values from each incoming
    :class:`~crappy.links.link.Link`, and returns the newest ones in a single
    dict.

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
    if not len(self._last_values):
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

    self.log(logging.DEBUG, f"Called recv_last_data, got data for labels "
                            f"{', '.join(ret.keys())}")
    return ret

  def recv_all_data(self) -> dict[str, list[Any]]:
    """Reads all values currently available on each incoming
    :class:`~crappy.links.link.Link` and merges them into a single dict.

    This method drains the values already queued on the Links and returns
    immediately. The returned :obj:`dict` might not always have a fixed number
    of keys, depending on the available data.

    Important:
      Values carrying the same label on different Links are concatenated in
      input-Link order. Their originating Link and chronological ordering
      across Links cannot be recovered from the result. Use
      :meth:`~crappy.blocks.meta_block.block.Block.recv_all_data_raw` when the
      Links must remain separate.

    Warning:
      A time label shared by several incoming Links is concatenated in the same
      way, so its values cannot reliably be associated with the other labels.

    Returns:
      A :obj:`dict` whose keys are the received labels and with a :obj:`list`
      of received values for each key. Values from a single Link remain ordered
      from oldest to newest. Values merged from several Links are grouped in
      input-Link order as described above.

    .. versionremoved:: 1.5.10 *num* argument
    .. versionadded:: 1.5.10 *blocking* argument
    .. versionremoved:: 2.0.0 *blocking* argument
    .. versionchanged:: 2.0.0 renamed from *get_all_last* to *recv_all_data*
    .. versionchanged:: 2.0.9 add new mechanism to avoid oversleeping
    .. versionremoved:: 2.1.0 *delay* and *poll_delay* arguments
    """

    ret = defaultdict(list)

    # Receiving from all input links
    for link in self.inputs:
      for label, values in link.recv_chunk().items():
        ret[label].extend(values)

    # Returning a dict, not a defaultdict
    self.log(logging.DEBUG, f"Called recv_all_data, got data for labels "
                            f"{', '.join(ret.keys())}")
    return dict(ret)

  def recv_all_data_raw(self) -> list[dict[str, list[Any]]]:
    """Reads all values currently available on each incoming
    :class:`~crappy.links.link.Link` and returns them separately.

    Unlike :meth:`~crappy.blocks.meta_block.block.Block.recv_all_data`, this
    method preserves one :obj:`dict` per Link, so the originating Link and its
    per-Link value ordering remain available.

    Returns:
      A :obj:`list` containing :obj:`dict`, whose keys are the received labels
      and with a :obj:`list` of received values for each key.
    
    .. versionadded:: 2.0.0
    .. versionchanged:: 2.0.9 add new mechanism to avoid oversleeping
    .. versionremoved:: 2.1.0 *delay* and *poll_delay* arguments
    """

    ret = [defaultdict(list) for _ in self.inputs]

    # Receiving from all input links
    for dic, link in zip(ret, self.inputs):
      dic |= link.recv_chunk()

    self.log(logging.DEBUG, f"Called recv_all_data_raw, got data for the "
                            f"following labels in the successive Links: "
                            f"{[', '.join(dic.keys()) for dic in ret]}")
    return [dict(dic) for dic in ret]

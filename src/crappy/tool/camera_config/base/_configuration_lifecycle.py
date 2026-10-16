# coding: utf-8

"""GUI-independent resources and failures for camera configuration sessions."""

from multiprocessing import synchronize
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from threading import Thread, Event as ThreadEvent

from ...._failure import FailureManager
from ...._exceptions import ProcessExitError, ShutdownTimeoutError


class ConfigurationLifecycle:
  """Owns GUI-independent resources and failures for a camera configuration
  session.
  """

  def __init__(self,
               stop_event: synchronize.Event,
               histogram_process: BaseProcess,
               queues: tuple[Queue, Queue],
               manager: FailureManager,
               failure_queue: Queue) -> None:
    """Initializes the resources and state for one configuration session.

    Args:
      stop_event: :obj:`Event <multiprocessing.Event>` shared with the
        histogram process to signal when the configuration window enters the
        closing phase.
      histogram_process: :obj:`Process <multiprocessing.Process>` computing
        preview histograms in parallel to the execution of the configuration
        window.
      queues: Queues communicating with the histogram process and that need to
        be closed when exiting the configuration window.
      manager: Configuration's manager, received from the owner of this class.
      failure_queue: Dedicated Queue for histogram failures.
    """

    # Objects shared by all configuration backends
    self.manager: FailureManager = manager
    self._stop_event: synchronize.Event = stop_event
    self._histogram_process: BaseProcess = histogram_process
    self._queues: tuple[Queue, Queue] = queues
    self._failure_queue: Queue = failure_queue
    self._receiver: Thread | None = None
    self._receiver_stop: ThreadEvent = ThreadEvent()
    self._histogram_identity: tuple[str, int | None]
    self._unexpected_exit: tuple[str, int | None, int] | None = None
    self._histogram_started: bool = False
    self._process_closed: bool = False
    self._stop_requested: bool = False
    self._cancelled_queues: list[Queue] = list()
    self._closed_queues: list[Queue] = list()
    self._closed: bool = False

  @property
  def closed(self) -> bool:
    """Whether resource cleanup has already begun."""

    return self._closed

  @property
  def failed(self) -> bool:
    """Whether the GUI Thread should close after a worker failure."""

    return bool(self.manager.snapshot() or self._unexpected_exit is not None)

  def mark_histogram_started(self) -> None:
    """Starts histogram reception after the histogram Process has successfully
    started."""

    self._histogram_started = True
    # Create and start the Thread for receiving failures from the Worker
    self._histogram_identity = (self._histogram_process.name,
                                self._histogram_process.pid)
    self._receiver = Thread(target=self._receive_failures,
                            name='histogram.failure-reader',
                            daemon=True)
    self._receiver.start()

  def _receive_failures(self) -> None:
    """Receives failure reports from the reporting Thread."""

    # Guard Exceptions with a FailureManager
    manager = self.manager
    queue = self._failure_queue
    stop = self._receiver_stop
    process = self._histogram_process
    identity = self._histogram_identity

    # Guard Exceptions with a FailureManager
    with manager.collect('Histogram reception', raise_on_exit=False):
      with manager.step('receive reports'):
        # Receive reports from the Worker
        while True:
          received = manager.receive((queue,))
          if not self._closed and self._unexpected_exit is None:
            # Check if the Worker stopped abruptly
            try:
              code = process.exitcode
            except ValueError:
              if self._closed:
                continue
              raise
            if isinstance(code, int):
              self._unexpected_exit = (*identity, code)
          # Only stop once there are no more reports to receive
          if stop.is_set() and not received:
            break
          if not received:
            stop.wait(0.05)

  def close_resources(self) -> None:
    """Stops the histogram and drains its reports before releasing Queues."""

    manager = self.manager
    with manager.collect('Configuration cleanup'):
      self._closed = True
      if not self._stop_requested:
        # Start by requesting everyone to stop
        with manager.step('request histogram stop'):
          self._stop_event.set()
          self._stop_requested = True

      # Check if the histogram Process is still running
      process = self._histogram_process
      terminated = False
      exit_info = self._unexpected_exit
      if not self._process_closed:
        if self._histogram_started:
          # Give the histogram process two seconds to stop
          with manager.step('join histogram'):
            process.join(2.0)
          alive = True
          # Check if the histogram Process stopped as expected
          with manager.step('check histogram'):
            alive = process.is_alive()
          if alive:
            # If it didn't, record the failure
            with manager.step('histogram shutdown deadline'):
              raise ShutdownTimeoutError("Histogram worker did not stop "
                                         "within 2 seconds",
                                         process_name=process.name,
                                         pid=process.pid,
                                         phase='configuration cleanup',
                                         timeout=2.0)
            # Still try to terminate the histogram Process
            with manager.step('terminate histogram'):
              process.terminate()
              terminated = True
            # Give the histogram Process half a second to terminate
            with manager.step('join terminated histogram'):
              process.join(0.5)
            alive = True
            # Check if the histogram Process is still alive after termination
            with manager.step('check terminated histogram'):
              alive = process.is_alive()
            if alive:
              # Still try to kill the histogram Process
              with manager.step('kill histogram'):
                process.kill()
                terminated = True
              # Give the histogram Process half a second to be killed
              with manager.step('join killed histogram'):
                process.join(0.5)
          # Report any unexpected exit code from the histogram Process
          with manager.step('observe histogram exit'):
            code = process.exitcode
            if isinstance(code, int) and code != 0 and exit_info is None:
              exit_info = (process.name, process.pid, code)
        # In all cases, close the histogram Process
        with manager.step('close histogram process'):
          process.close()
          self._process_closed = True

      # Stop the Thread receiving Exceptions from the histogram Process
      with manager.step('stop histogram receiver'):
        self._receiver_stop.set()
      # Give the Thread half a second to stop
      with manager.step('join histogram receiver'):
        if self._receiver is not None and self._receiver.ident is not None:
          self._receiver.join(0.5)
          # Report it if the Thread is still alive
          if self._receiver.is_alive():
            raise ShutdownTimeoutError("Histogram failure receiver did not "
                                       "stop", phase='monitoring', timeout=0.5)
          self._receiver = None

      # Close the Queue carrying Exceptions
      if self._failure_queue not in self._closed_queues:
        with manager.step('close histogram failure Queue'):
          self._failure_queue.close()
          self._closed_queues.append(self._failure_queue)

      # Record and report an unexpected exit code
      if exit_info is not None and not terminated:
        name, pid, code = exit_info
        reported = {(report.process_name, report.pid)
                    for report, _ in manager.snapshot()}
        reported.update((error.process_name, error.pid)
                        for _, error in manager.snapshot()
                        if isinstance(error, ProcessExitError))
        with manager.step('verify histogram exit'):
          if (name, pid) not in reported:
            raise ProcessExitError(f"Histogram worker {name} exited "
                                   f"unexpectedly ({code})",
                                   process_name=name,
                                   pid=pid,
                                   phase='configuration',
                                   exitcode=code)

      # Close the various Queues that were opened
      for index, queue in enumerate(self._queues):
        if queue not in self._cancelled_queues:
          with manager.step(f'cancel preview Queue {index + 1} feeder join'):
            queue.cancel_join_thread()
            self._cancelled_queues.append(queue)
        if queue not in self._closed_queues:
          with manager.step(f'close preview Queue {index + 1}'):
            queue.close()
            self._closed_queues.append(queue)

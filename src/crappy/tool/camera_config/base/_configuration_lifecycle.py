# coding: utf-8

"""GUI-independent resources and failures for camera configuration sessions."""

from collections.abc import Callable, Iterable
import logging
from multiprocessing import synchronize
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from types import TracebackType

ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]


class ConfigurationLifecycle:
  """Owns GUI-independent resources and failures for a camera configuration
  session.
  """

  def __init__(self,
               stop_event: synchronize.Event,
               histogram_process: BaseProcess,
               queues: Iterable[Queue],
               log: Callable[..., None]) -> None:
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
      log: Method to use for logging information.
    """

    self._stop_event: synchronize.Event = stop_event
    self._histogram_process: BaseProcess = histogram_process
    self._queues: tuple[Queue, ...] = tuple(queues)
    self._log: Callable[..., None] = log
    self._histogram_started: bool = False
    self._process_closed: bool = False
    self._stop_requested: bool = False
    self._cancelled_queues: list[Queue] = list()
    self._closed_queues: list[Queue] = list()
    self._closed: bool = False
    self._failure: BaseException | None = None
    self._failure_traceback: TracebackType | None = None

  @property
  def closed(self) -> bool:
    """Whether resource cleanup has already begun."""

    return self._closed

  def mark_histogram_started(self) -> None:
    """Called to indicated that the histogram process has started, and that it
    can therefore be joined if needed."""

    self._histogram_started = True

  def record_callback_failure(self,
                              error: BaseException,
                              traceback: TracebackType | None) -> None:
    """Retains the first callback error for re-raising after the GUI closes.

    Keyboard interrupts are retained without an error log.

    Args:
      error: Exception raised by a toolkit callback.
      traceback: Original callback traceback, or :obj:`None` if unavailable.
    """

    if self._failure is None:
      self._failure = error
      self._failure_traceback = traceback

    if not isinstance(error, KeyboardInterrupt):
      self._log(logging.ERROR, "Configuration callback failed",
                exc_info=(type(error), error, traceback))

  def raise_if_failed(self) -> None:
    """Re-raises a callback failure hidden by the GUI toolkit's event loop."""

    if self._failure is not None:
      raise self._failure.with_traceback(self._failure_traceback)

  def request_close(self, close: Callable[[], None]) -> None:
    """Asks a GUI backend to close after retaining a callback failure.

    Errors from closing are logged without replacing the original failure.

    Args:
      close: Backend method that closes the window without validation.
    """

    try:
      close()
    except Exception as exc:
      self._log(logging.ERROR, "Could not close configuration window",
                exc_info=(type(exc), exc, exc.__traceback__))

  def close_resources(self) -> None:
    """Stops the histogram process and queues once, including before start.

    Requests graceful worker shutdown, then terminates or kills the process if
    it remains alive. Queue cleanup is attempted even if process cleanup fails.
    """

    failures: list[Exception | KeyboardInterrupt] = list()
    self._closed = True

    # Make sure that the stop event is set at that point
    if not self._stop_requested:
      try:
        self._stop_event.set()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note("Configuration cleanup step: request histogram stop")
        failures.append(error)
      else:
        self._stop_requested = True

    process = self._histogram_process
    if not self._process_closed:
      if self._histogram_started:
        # Give the histogram process a chance to stop on its own
        try:
          process.join(1.0)
        except (Exception, KeyboardInterrupt) as error:
          error.add_note("Configuration cleanup step: join histogram "
                         "(initial wait)")
          failures.append(error)

        # Terminate the process if it did not stop
        try:
          alive = process.is_alive()
        except (Exception, KeyboardInterrupt) as error:
          error.add_note("Configuration cleanup step: check histogram")
          failures.append(error)
          alive = True
        if alive:
          try:
            process.terminate()
          except (Exception, KeyboardInterrupt) as error:
            error.add_note("Configuration cleanup step: terminate histogram")
            failures.append(error)

          # Give the histogram process some time to stop after termination
          try:
            process.join(1.0)
          except (Exception, KeyboardInterrupt) as error:
            error.add_note("Configuration cleanup step: join histogram "
                           "(after terminate)")
            failures.append(error)

          # Kill the process if termination did not stop it
          try:
            alive = process.is_alive()
          except (Exception, KeyboardInterrupt) as error:
            error.add_note("Configuration cleanup step: check histogram")
            failures.append(error)
            alive = True
          if alive:
            try:
              process.kill()
            except (Exception, KeyboardInterrupt) as error:
              error.add_note("Configuration cleanup step: kill histogram")
              failures.append(error)

            # Give the histogram process some time to stop after killing
            try:
              process.join(1.0)
            except (Exception, KeyboardInterrupt) as error:
              error.add_note("Configuration cleanup step: join histogram "
                             "(after kill)")
              failures.append(error)

      # Ultimately close the histogram process
      try:
        process.close()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note("Configuration cleanup step: close histogram process")
        failures.append(error)
      else:
        self._process_closed = True

    # Cancel feeder-thread joins before closing each owned Queue
    for index, queue in enumerate(self._queues):
      if queue not in self._cancelled_queues:
        try:
          queue.cancel_join_thread()
        except (Exception, KeyboardInterrupt) as error:
          error.add_note(f"Configuration cleanup step: cancel Queue "
                         f"{index + 1} feeder join")
          failures.append(error)
        else:
          self._cancelled_queues.append(queue)

      # Actually close the Queues
      if queue not in self._closed_queues:
        try:
          queue.close()
        except (Exception, KeyboardInterrupt) as error:
          error.add_note(f"Configuration cleanup step: close Queue {index + 1}")
          failures.append(error)
        else:
          self._closed_queues.append(queue)

    # If there's only one Exception, raise it
    if len(failures) == 1:
      raise failures[0]
    # Handle the case when a KeyboardInterrupt is among the Exceptions
    elif any(isinstance(error, KeyboardInterrupt) for error in failures):
      for index, error in enumerate(failures):
        if isinstance(error, KeyboardInterrupt):
          others: list[BaseException] = failures[:index] + failures[index + 1:]
          if error.__cause__ is not None:
            others.insert(0, error.__cause__)
          raise error from BaseExceptionGroup("Other Configuration cleanup "
                                              "failures", others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("Configuration cleanup failures", failures)

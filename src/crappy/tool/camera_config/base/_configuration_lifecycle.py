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
    self._queues: tuple[Queue] = tuple(queues)
    self._log: Callable[..., None] = log
    self._histogram_started: bool = False
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

    # Nothing to do if the window was already closed
    if self._closed:
      return

    # Request the histogram to stop
    self._closed = True
    self._stop_event.set()

    try:
      process = self._histogram_process
      # First, check if the histogram process stopped on its own
      if self._histogram_started or process.is_alive():
        process.join(1.0)
        # If not, try to terminate it
        if process.is_alive():
          self._log(logging.WARNING, "The histogram process did not stop "
                                     "within the timeout, terminating it")
          process.terminate()
          process.join(1.0)
        # If still alive, now try to kill it
        if process.is_alive():
          self._log(logging.WARNING, "The histogram process did not terminate "
                                     "within the timeout, killing it")
          process.kill()
          process.join()

    # Unconditionally clean up the queue resources
    finally:
      for queue in self._queues:
        try:
          queue.cancel_join_thread()
        except Exception as exc:
          self._log(logging.ERROR, "Could not cancel histogram queue thread "
                                   "join",
                    exc_info=(type(exc), exc, exc.__traceback__))
        try:
          queue.close()
        except Exception as exc:
          self._log(logging.ERROR, "Could not close histogram queue",
                    exc_info=(type(exc), exc, exc.__traceback__))

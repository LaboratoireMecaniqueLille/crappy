# coding: utf-8

"""Backend-independent configurator contract and resource lifetime.

The owning Block uses this contract regardless of the GUI backend.
"""

from collections.abc import Callable, Iterable
import logging
from multiprocessing import synchronize
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from types import TracebackType
from typing import Any, Protocol
import numpy as np

from ....camera.meta_camera import Camera

ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]


class CameraConfigurator(Protocol):
  """Lifecycle and output required by Camera and Vision Block callers.

  GUI implementations may use any widget toolkit. ``run()`` owns the complete
  interactive lifetime and raises a retained callback failure after closing.
  ``stop()`` is safe to call again when a caller handles an exception.

  Every backend implements ``watch_shutdown(predicate)`` to close its UI when
  the owning Block stops or the preparation Barrier breaks. Block callers check
  the predicate again after ``run()`` before reading any configuration output.
  """

  shape: tuple[int, int] | tuple[int, int, int] | None
  dtype: str | None

  def run(self) -> None:
    """Run the configuration GUI until it closes or fails."""

  def stop(self) -> None:
    """Release resources and close the UI, including after a partial start."""

  def watch_shutdown(self, requested: Callable[[], bool]) -> None:
    """Close without validation when Block preparation must stop."""

  def get_config(self) -> tuple[Any, ...] | None:
    """Return values consumed by the paired camera processing Block."""


ConfiguratorFactory = Callable[..., CameraConfigurator]


def create_configurator(configurator: ConfiguratorFactory,
                        camera: Camera,
                        config_backend: str,
                        log_queue: Queue,
                        log_level: int | None,
                        max_freq: float | None,
                        transform: Callable[[np.ndarray], np.ndarray] | None,
                        *args: Any,
                        **kwargs: Any) -> CameraConfigurator:
  """Instantiate a configurator with arguments shared by both Block paths.

  The public Camera Blocks validate ``config_backend``. Tk uses the supplied
  configurator class, including explicit custom classes. The PyQt option is
  reserved until its configurator classes are implemented.
  """

  if config_backend == 'tkinter':
    return configurator(camera, log_queue, log_level, max_freq, transform,
                        *args, **kwargs)
  elif config_backend == 'pyqt':
    raise NotImplementedError("The 'pyqt' CameraConfig backend is not yet "
                              "implemented")
  else:
    raise ValueError(f"Unknown config_backend: {config_backend!r}")


class ConfigurationLifecycle:
  """Own non-GUI resources and failures across a configuration session.

  Args:
    stop_event: Event shared with the histogram process.
    histogram_process: Process computing preview histograms.
    queues: Queues communicating with that process.
    log: Configurator logging callback, including exception information.
  """

  def __init__(self,
               stop_event: synchronize.Event,
               histogram_process: BaseProcess,
               queues: Iterable[Queue],
               log: Callable[..., None]) -> None:

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
    """Record that joining the histogram process is now valid."""

    self._histogram_started = True

  def record_callback_failure(self,
                              error: BaseException,
                              traceback: TracebackType | None) -> None:
    """Retain the first callback error for re-raising after the UI closes."""

    if self._failure is None:
      self._failure = error
      self._failure_traceback = traceback

    self._log(logging.ERROR, "Configuration callback failed",
              exc_info=(type(error), error, traceback))

  def raise_if_failed(self) -> None:
    """Expose a callback failure hidden by the GUI toolkit's event loop."""

    if self._failure is not None:
      raise self._failure.with_traceback(self._failure_traceback)

  def request_close(self, close: Callable[[], None]) -> None:
    """Ask a GUI backend to close after retaining a callback failure."""

    try:
      close()
    except Exception as exc:
      self._log(logging.ERROR, "Could not close configuration UI",
                exc_info=(type(exc), exc, exc.__traceback__))

  def close_resources(self) -> None:
    """Stop the histogram process and queues once, including before start."""

    if self._closed:
      return

    self._closed = True
    self._stop_event.set()

    try:
      process = self._histogram_process
      # Process.join() raises AssertionError before Process.start()
      if self._histogram_started or process.is_alive():
        process.join(1.0)
        if process.is_alive():
          self._log(logging.WARNING, "The histogram process failed to stop, "
                                     "terminating it !")
          process.terminate()
          process.join(1.0)
        if process.is_alive():
          self._log(logging.WARNING, "The histogram process failed to "
                                     "terminate, killing it !")
          process.kill()
          process.join()

    finally:
      for queue in self._queues:
        try:
          queue.cancel_join_thread()
        except Exception as exc:
          self._log(logging.ERROR, "Could not join thread of histogram queue",
                    exc_info=(type(exc), exc, exc.__traceback__))
        try:
          queue.close()
        except Exception as exc:
          self._log(logging.ERROR, "Could not close histogram queue",
                    exc_info=(type(exc), exc, exc.__traceback__))

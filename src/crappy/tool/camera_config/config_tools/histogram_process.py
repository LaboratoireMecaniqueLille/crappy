# coding: utf-8

import numpy as np
from multiprocessing import Process, current_process, get_start_method
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
import logging
import logging.handlers
from functools import partial

from ...._failure import FailureManager
from ...._exceptions import StopRequested


class HistogramProcess(Process):
  """Worker process that computes an 8-bit preview histogram through queues.

  Concrete camera configuration windows submit reduced grayscale images and
  receive histogram images independently of their GUI event loop. The worker
  processes the newest queued request and uses a shared event to report
  activity.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               stop_event: Event,
               processing_event: Event,
               img_in: Queue,
               img_out: Queue,
               log_level: int | None,
               log_queue: Queue,
               failure_queue: Queue,
               block_name: str,
               owner_path: str) -> None:
    """Initializes worker resources without starting the process.

    Args:
      stop_event: Shared :obj:`multiprocessing.Event` requesting worker
        shutdown.
      processing_event: Shared :obj:`multiprocessing.Event` set while a request
        is being processed.
      img_in: Input :obj:`~multiprocessing.Queue` carrying (image, auto_range,
        low_thresh, high_thresh) tuples. Images contain 8-bit grayscale preview
        pixels.
      img_out: Output :obj:`~multiprocessing.Queue` receiving uint8 histogram
        images of shape (80, 512). Background pixels are 255, bars are 0, and
        Auto range markers are 127.
      log_level: Worker logging level, or :obj:`None` to disable worker
        logging.
      log_queue: Crappy logging queue, used with spawn and forkserver start
        methods to forward records to the main process.
      failure_queue: :obj:`~multiprocessing.Queue` for reporting failures to
        the configuration window.

        .. versionadded:: 2.2.0
      block_name: Name of the parent Block, shown in failure reports.

        .. versionadded:: 2.2.0
      owner_path: Name identifying this worker within its parent Block.

        .. versionadded:: 2.2.0
    """

    self._logger: logging.Logger | None = None
    self._log_level = log_level
    self._log_queue = log_queue

    super().__init__(name=f"{current_process().name}.{type(self).__name__}")

    # Attributes for managing failure reporting
    self._failure_queue = failure_queue
    self._failure_manager: FailureManager
    self._block_name: str = block_name
    self._owner_path: str = owner_path

    self._stop_event: Event = stop_event
    self._processing_event: Event = processing_event
    self._img_in: Queue = img_in
    self._img_out: Queue = img_out

  def run(self) -> None:
    """Processes histogram requests until shutdown or failure.

    Calculates 256 intensity bins, scales the largest bin to the display
    height, and adds optional Auto range markers. Results are images for
    backend rendering, not numerical bin counts.
    """

    # Make sure there's a FailureManager running
    self._failure_manager = FailureManager(self._block_name,
                                           'bootstrap',
                                           owner_path=self._owner_path)
    manager = self._failure_manager

    # Guard execution with a FailureManager
    with manager.collect('Histogram lifecycle', raise_on_exit=False):
      with manager.step('execution'):
        # Start with configuring the FailureManager and logger
        manager.set_queue(self._failure_queue)
        self._set_logger()
        if self._logger is not None:
          manager._logger = self._logger
        manager.phase = 'configuration'

        self._processing_event.clear()
        self.log(logging.DEBUG, "Histogram worker started")

        # Looping until told to stop or an exception is raised
        latest = None
        while not self._stop_event.is_set():

          # Setting the processing event when busy processing an image
          try:
            # Add timeout to avoid spamming the CPU when no image is available
            latest = self._img_in.get(timeout=0.01)
            self._processing_event.set()
            while True:
              latest = self._img_in.get_nowait()
          except Empty:
            if latest is None:
              continue
          # Case when reading from the Queue fails
          except (EOFError, OSError):
            if self._stop_event.is_set():
              raise StopRequested from None
            raise

          img, auto_range, low_thresh, high_thresh = latest

          # Fast-forward without running calculation
          if self._stop_event.is_set():
            break

          self.log(logging.DEBUG, "Received image from CameraConfig")

          # Calculating the histogram
          hist, _ = np.histogram(img, bins=np.arange(257))
          hist = np.repeat(hist / np.max(hist) * 80, 2)
          hist = np.repeat(hist[np.newaxis, :], 80, axis=0)

          # Making a nice image out of the calculated histogram
          out_img = np.fromfunction(partial(self._hist_func, histo=hist),
                                    shape=(80, 512))
          out_img = np.flip(out_img, axis=0).astype('uint8')

          # Adding vertical grey bars to indicate the limits of the
          # auto-range
          if auto_range:
            self.log(logging.DEBUG, "Drawing Auto range threshold "
                                    "markers")
            out_img[:, round(2 * low_thresh)] = 127
            out_img[:, round(2 * high_thresh)] = 127

          # Sending back the histogram
          self._img_out.put_nowait(out_img)
          self.log(logging.DEBUG, "Sent the histogram back to the "
                                  "CameraConfig")
          latest = None
          self._processing_event.clear()

        self.log(logging.DEBUG, "Histogram worker stopping after shutdown "
                                "request")
      manager.phase = 'configuration cleanup'

      # Clean up the managed multiprocessing objects
      with manager.step('clear processing Event'):
        self._processing_event.clear()
      with manager.step('release preview Queues'):
        self._cleanup_queues()
      manager.phase = 'failure Queue shutdown'

      # Handle proper termination of the FailureManager
      with manager.step('disconnect'):
        manager.set_queue(None)
      with manager.step('close'):
        self._failure_queue.close()
      with manager.step('flush'):
        self._failure_queue.join_thread()

  def _cleanup_queues(self) -> None:
    """Drains and releases both worker Queue handles, even after failure."""

    manager = self._failure_manager

    # Flush and clean up the managed Queues
    with manager.collect('Histogram preview cleanup'):
      for index, queue in enumerate((self._img_in, self._img_out)):
        with manager.step(f'drain Queue {index + 1}'):
          try:
            while True:
              queue.get_nowait()
          except Empty:
            pass
        with manager.step(f'cancel Queue {index + 1} feeder join'):
          queue.cancel_join_thread()
        with manager.step(f'close Queue {index + 1}'):
          queue.close()

  @staticmethod
  def _hist_func(x: np.ndarray,
                 _: np.ndarray,
                 histo: np.ndarray) -> np.ndarray:
    """Function passed to the :meth:`numpy.fromfunction` method for building
    the histogram."""

    return np.where(x <= histo, 0, 255)

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the
    :class:`~crappy.tool.camera_config.config_tools.HistogramProcess`.

    Also instantiates the :obj:`~logging.Logger` when logging the first
    message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._set_logger()

    self._logger.log(level, msg)

  def _set_logger(self) -> None:
    """Instantiates and sets up the logger for the
    :class:`~crappy.tool.camera_config.config_tools.HistogramProcess`."""

    logger = logging.getLogger(self.name)

    # Disabling logging if requested
    logger.disabled = self._log_level is None
    if self._log_level is not None:
      logger.setLevel(self._log_level)

    # On spawn and forkserver, the messages need to be sent through a Queue for
    # logging
    if get_start_method() != 'fork' and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger

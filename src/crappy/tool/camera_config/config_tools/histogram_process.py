# coding: utf-8

import numpy as np
from multiprocessing import Process, current_process, get_start_method
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
import logging
import logging.handlers
from functools import partial


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
               log_queue: Queue) -> None:
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
    """

    self._logger: logging.Logger | None = None
    self._log_level = log_level
    self._log_queue = log_queue

    super().__init__(name=f"{current_process().name}.{type(self).__name__}")

    self._stop_event: Event = stop_event
    self._processing_event: Event = processing_event
    self._img_in: Queue = img_in
    self._img_out: Queue = img_out
    self._drained_queues: list[Queue] = list()
    self._cancelled_queues: list[Queue] = list()
    self._closed_queues: list[Queue] = list()

  def run(self) -> None:
    """Processes histogram requests until shutdown or failure.

    Calculates 256 intensity bins, scales the largest bin to the display
    height, and adds optional Auto range markers. Results are images for
    backend rendering, not numerical bin counts.
    """

    try:
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

        try:
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

          # Adding vertical grey bars to indicate the limits of the auto range
          if auto_range:
            self.log(logging.DEBUG, "Drawing Auto range threshold markers")
            out_img[:, round(2 * low_thresh)] = 127
            out_img[:, round(2 * high_thresh)] = 127

          # Sending back the histogram
          self._img_out.put_nowait(out_img)
          self.log(logging.DEBUG, "Sent the histogram back to the "
                                  "CameraConfig")

        # Cleanup performed at the end of each processing
        finally:
          latest = None
          self._processing_event.clear()

      self.log(logging.DEBUG, "Histogram worker stopping after shutdown "
                              "request")

    except KeyboardInterrupt:
      self.log(logging.DEBUG, "Histogram worker interrupted")
    except (Exception,) as exc:
      if self._logger is None:
        self._set_logger()
      self._logger.exception("Histogram processing failed",
                             exc_info=exc)
    finally:
      self._cleanup_queues()

  def _cleanup_queues(self) -> None:
    """Drains and releases both worker Queue handles, even after failure."""

    failures: list[Exception | KeyboardInterrupt] = list()

    for index, queue in enumerate((self._img_in, self._img_out)):
      if (queue not in self._drained_queues and
          queue not in self._closed_queues):

        # First, flush the Queues
        try:
          self._flush_queue(queue)
        except (Exception, KeyboardInterrupt) as error:
          error.add_note(f"HistogramProcess cleanup step: drain Queue "
                         f"{index + 1}")
          failures.append(error)
        else:
          self._drained_queues.append(queue)

      # Then, cancel the feeder join
      if queue not in self._cancelled_queues:
        try:
          queue.cancel_join_thread()
        except (Exception, KeyboardInterrupt) as error:
          error.add_note(f"HistogramProcess cleanup step: cancel Queue "
                         f"{index + 1} feeder join")
          failures.append(error)
        else:
          self._cancelled_queues.append(queue)

      # Then, close the queues
      if queue not in self._closed_queues:
        try:
          queue.close()
        except (Exception, KeyboardInterrupt) as error:
          error.add_note(f"HistogramProcess cleanup step: close Queue "
                         f"{index + 1}")
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
          raise error from BaseExceptionGroup("Other HistogramProcess cleanup "
                                              "failures", others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("HistogramProcess cleanup failures", failures)

  @staticmethod
  def _hist_func(x: np.ndarray,
                 _: np.ndarray,
                 histo: np.ndarray) -> np.ndarray:
    """Function passed to the :meth:`numpy.fromfunction` method for building
    the histogram."""

    return np.where(x <= histo, 0, 255)

  @staticmethod
  def _flush_queue(queue: Queue) -> None:
    """Drains a :obj:`~multiprocessing.Queue` before exiting.

    Pending queue contents can prevent the
    :class:`~crappy.tool.camera_config.config_tools.HistogramProcess` from
    finishing on time, especially with the spawn start method.
    """

    try:
      while True:
        queue.get_nowait()
    except Empty:
      pass

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
    if self._log_level is not None:
      logger.setLevel(self._log_level)
    else:
      logging.disable()

    # On spawn and forkserver, the messages need to be sent through a Queue for
    # logging
    if get_start_method() != 'fork' and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger

# coding: utf-8

from multiprocessing import Process, get_start_method
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import Event

import numpy as np
from time import monotonic
from select import select
import logging
import logging.handlers
from platform import system
from pickle import loads
from multiprocessing import synchronize

from ...._failure import FailureManager
from ...._exceptions import StopRequested

from ...camera_config import Box
from ...._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
try:
  from skimage.filters import threshold_otsu
except (ModuleNotFoundError, ImportError):
  threshold_otsu = OptionalModule("skimage", "Please install scikit-image to "
                                             "use Video-extenso")


class LostSpotError(Exception):
  """Exception raised when a spot is lost, or when there's too much
  overlapping."""


class Tracker(Process):
  """:obj:`multiprocessing.Process` whose task is to track a spot on an image.

  It receives a subframe centered on the last known position of the spot, and
  returns the updated position of the detected spot. It is meant to be used in
  association with the 
  :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.
  
  .. versionadded:: 2.0.0
  """

  names = list()

  def __init__(self,
               pipe: Connection,
               logger_name: str,
               log_level: int | None,
               log_queue: Queue,
               stop_event: synchronize.Event,
               white_spots: bool = False,
               thresh: int | None = None,
               blur: int | None = 5,
               failure_queue: Queue | None = None,
               block_name: str | None = None,
               raise_on_lost_spot: bool = False) -> None:
    """Sets the arguments.

    Args:
      pipe: The :obj:`~multiprocessing.connection.Connection` object through
        which the image is received and the updated coordinates of the spot
        are sent back.
      logger_name: The name of the parent :obj:`~logging.Logger` as a 
        :obj:`str`, used for naming the Logger in this class.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.
      stop_event: Shared Event for requesting shutdown.
      white_spots: If :obj:`True`, detects white objects on a black background,
        else black objects on a white background.
      thresh: If given, this threshold value will always be used for isolating
        the spot from the background. If not given (:obj:`None`), a new
        threshold is recalculated for each new subframe. Spots are less likely
        to be lost with an adaptive threshold, but it takes a bit more time.
      failure_queue: Queue for reporting unexpected failures.
      block_name: Name of the parent Block.
      raise_on_lost_spot: Whether to report a lost spot as a failure or to
        silently ignore it.
      blur: If not :obj:`None`, the subframe is first blurred before trying to
        detect the spot. This argument gives the size of the kernel to use for
        blurring. Better results are obtained with blurring, but it takes a bit
        more time.
    """

    super().__init__()
    self.name = self.get_name(logger_name, type(self).__name__)
    self._system = system()

    if blur is not None and (blur < 1
                             or not blur % 2
                             or not isinstance(blur, int)):
      raise ValueError("blur must be a positive odd integer, or None")

    # Attributes for failure handling and report
    self._failure_queue: Queue | None = failure_queue
    self._failure_manager: FailureManager | None = None
    self._stop_event: synchronize.Event = stop_event
    self._block_name: str = block_name or logger_name
    self._raise_on_lost_spot: bool = raise_on_lost_spot

    self._pipe = pipe
    self._white_spots = white_spots
    self._thresh = thresh
    self._blur = blur

    self._logger: logging.Logger | None = None
    self._log_level = log_level
    self._log_queue = log_queue

    self._n = 0
    self._last_warn = monotonic()

  @classmethod
  def get_name(cls, logger_name: str, self_name: str) -> str:
    """Method for naming the Tracker processes so that each of them has a
    unique name.

    Args:
      logger_name: The name of the parent :obj:`~logging.Logger`, as a 
        :obj:`str`.
      self_name: The name of the current class.

    Returns:
      The chosen name for the current Tracker process, as a :obj:`str`.
    """

    i = 1
    while f"{logger_name}.{self_name}-{i}" in cls.names:
      i += 1

    cls.names.append(f"{logger_name}.{self_name}-{i}")
    return f"{logger_name}.{self_name}-{i}"

  def run(self) -> None:
    """Continuously reads incoming subframes, tries to detect a spot and sends
    back the coordinates of the detected spot."""

    # Make sure there's a FailureManager running
    self._failure_manager = FailureManager(self._block_name,
                                           'bootstrap',
                                           owner_path=self.name)
    manager = self._failure_manager

    # Guard execution with a FailureManager
    with manager.collect('Tracker lifecycle', raise_on_exit=False):
      with manager.step('execution'):
        # Start with configuring the FailureManager and logger
        if self._failure_queue is not None:
          manager.set_queue(self._failure_queue)
        self._set_logger()
        if self._logger is not None:
          manager._logger = self._logger
        manager.phase = 'tracking'

        while not self._stop_event.is_set():
          try:
            # Making sure the call to recv is not blocking
            if not self._pipe.poll(0.1):
              continue
            y_start, x_start, img = loads(self._pipe.recv_bytes())
          # Case when the received values cannot be read
          except (EOFError, OSError):
            if self._stop_event.is_set():
              raise StopRequested from None
            raise
          self._log(logging.DEBUG, "Received data from pipe")
          self._n += 1

          # Evaluate the new position of the spot and handle a lost spot
          try:
            box = self._evaluate(x_start, y_start, img)
          except LostSpotError as error:
            if self._raise_on_lost_spot:
              raise
            self._pipe.send(str(error) or 'The selected spot was lost')
            break

          # Simply sending back the new Box containing the spot
          self._send(box)

      # Clean up the multiprocessing resources owned by the Process
      manager.phase = 'tracker cleanup'
      with manager.step('close tracking Pipe'):
        self._pipe.close()
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

  def _evaluate(self, x_start: int, y_start: int, img: np.ndarray) -> Box:
    """Takes a sub-image, applies a threshold on it and tries to detect the new
    position of the spot.

    Args:
      x_start: The x position of the top left pixel of the subframe on the
        entire image.
      y_start: The y position of the top left pixel of the subframe on the
        entire image.
      img: The subframe on which to search for a spot.

    Returns:
      A :class:`~crappy.tool.camera_config.config_tools.Box` object containing 
      the x and y start and end positions of the detected spot, as well as the 
      coordinates of the centroid.
    """

    # First, blurring the image if asked to
    if self._blur is not None and self._blur > 1:
      img = cv2.medianBlur(img, self._blur)

    # Determining the best threshold for the image if required
    thresh = self._thresh if self._thresh is not None else threshold_otsu(img)

    # Getting all pixels superior or inferior to threshold
    if self._white_spots:
      black_white = (img > thresh).astype('uint8')
    else:
      black_white = (img <= thresh).astype('uint8')

    # Checking that the detected spot is large enough
    if np.count_nonzero(black_white) < 0.1 * img.size:

      # If the threshold is pre-defined, trying again with an updated one
      if self._thresh is not None:
        self._log(logging.WARNING,
                  "Detected spot too small compared with overall box size, "
                  "recalculating threshold")
        thresh = threshold_otsu(img)
        if self._white_spots:
          black_white = (img > thresh).astype('uint8')
        else:
          black_white = (img <= thresh).astype('uint8')

        # If the spot still cannot be detected, aborting
        if np.count_nonzero(black_white) < 0.1 * img.size:
          raise LostSpotError

      # If an adaptive threshold is already used, nothing more can be done
      else:
        raise LostSpotError

    # Calculating the coordinates of the centroid using the image moments
    moments = cv2.moments(black_white)
    try:
      x = moments['m10'] / moments['m00']
      y = moments['m01'] / moments['m00']
    except ZeroDivisionError:
      raise ZeroDivisionError("Couldn't compute the centroid because the "
                              "moment of order 0, 0 is zero !")

    # Getting the updated centroid and coordinates of the spot
    x_min, y_min, width, height = cv2.boundingRect(black_white)
    return Box(x_start=x_start + x_min,
               y_start=y_start + y_min,
               x_end=x_start + x_min + width,
               y_end=y_start + y_min + height,
               x_centroid=x_start + x,
               y_centroid=y_start + y)

  def _send(self, val: Box) -> None:
    """Sends a message to the 
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`, and 
    in Linux checks that the :obj:`multiprocessing.Pipe` is not full before 
    sending."""

    if self._system == 'Linux':
      if select([], [self._pipe], [], 0)[1]:
        self._pipe.send(val)
      elif monotonic() - self._last_warn > 1:
          self._last_warn = monotonic()
          self._log(logging.WARNING, f"Cannot send the detected spot to the "
                                     f"VideoExtenso tool, the Pipe is full !")
    else:
      self._pipe.send(val)

  def _set_logger(self) -> None:
    """Instantiates and sets up the logger for the instance."""

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

  def _log(self, level: int, msg: str) -> None:
    """Sends a log message to the :obj:`~logging.Logger`.

    Args:
      level: The logging level, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(level, msg)

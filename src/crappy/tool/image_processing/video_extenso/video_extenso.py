# coding: utf-8

from multiprocessing import Pipe, current_process
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from typing import Optional, Union
import numpy as np
from itertools import combinations
from time import sleep, time
import logging
import logging.handlers
from select import select
from platform import system

from ...camera_config import SpotsBoxes, Box
from .tracker import Tracker, LostSpotError


class VideoExtensoTool:
  """This class is the core of the :class:`~crappy.blocks.VideoExtenso` Block.

  It performs spot tracking on up to `4` spots on the images acquired by the
  :class:`~crappy.camera.Camera`, and computes the strain values at each new 
  image. For each spot, the tracking is performed by an independent
  :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker` Process.

  It is possible to track only one spot, in which case only the position of its
  center is returned and the strain values are left to `0`.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10 renamed from *Video_extenso* to *VideoExtenso*
  .. versionchanged:: 2.0.0 renamed from *VideoExtenso* to *VideoExtensoTool*
  """

  def __init__(self,
               spots: SpotsBoxes,
               thresh: int,
               log_level: Optional[int],
               log_queue: Queue,
               white_spots: bool = False,
               update_thresh: bool = False,
               safe_mode: bool = False,
               border: int = 5,
               blur: Optional[int] = 5) -> None:
    """Sets the arguments and the other instance attributes.

    Args:
      spots: An instance of the 
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` tool 
        containing the coordinates of the spots to track.

        .. versionadded:: 2.0.0
      thresh: The grey level value of the threshold to use for discriminating
        spots from the background, as an :obj:int`. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      white_spots: If :obj:`True`, detects white objects over a black
        background, else black objects over a white background. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      update_thresh: If :obj:`True`, the grey level threshold for detecting the
        spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      safe_mode: If :obj:`True`, the class will stop and raise an exception as
        soon as overlapping spots are detected. Otherwise, it will first try to
        reduce the detection window to get rid of overlapping. This argument
        should be used when inconsistency in the results may have critical
        consequences.
      border: When searching for the new position of a spot, the class will
        search in the last known bounding box of this spot plus a few
        additional pixels in each direction. This argument sets the number of
        additional pixels to use. It should be greater than the expected
        "speed" of the spots, in pixels / frame. But if it's set too high,
        noise or other spots might hinder the detection. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      blur: The size in pixels (as an odd :obj:`int` greater than `1`) of the
        kernel to use when applying a median blur filter to the image before
        the spot detection. If not given, no blurring is performed. A slight 
        blur improves the spot detection by smoothening the noise, but also 
        takes a bit more time compared to no blurring. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.

    .. versionremoved:: 2.0.0 *num_spots* and *min_area* arguments
    """

    # These attributes will be used later
    self._consecutive_overlaps = 0
    self._trackers = list()
    self._pipes = list()

    # Setting the args
    self._white_spots = white_spots
    self._update_thresh = update_thresh
    self._safe_mode = safe_mode
    self._border = border
    self._blur = blur
    self.spots = spots
    self._thresh = thresh

    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._log_queue = log_queue

    self._last_warn = time()
    self._system = system()

  def __del__(self) -> None:
    """Security to ensure there are no zombie processes left when exiting."""

    self.stop_tracking()

  def start_tracking(self) -> None:
    """Creates a
    :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
    Process for each detected spot, and starts it.

    Also creates a :obj:`multiprocessing.Pipe` for each spot to communicate 
    with the Tracker process.
    """

    if self.spots.empty():
      raise AttributeError("No spots selected, aborting !")

    for spot in self.spots:
      if spot is None:
        continue

      inlet, outlet = Pipe()
      tracker = Tracker(pipe=outlet,
                        logger_name=f"{current_process().name}."
                                    f"{type(self).__name__}",
                        log_level=self._log_level,
                        log_queue=self._log_queue,
                        white_spots=self._white_spots,
                        thresh=None if self._update_thresh else self._thresh,
                        blur=self._blur)
      self._pipes.append(inlet)
      self._trackers.append(tracker)
      tracker.start()

  def stop_tracking(self) -> None:
    """Stops all the active 
    :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
    Processes, either gently or by terminating them if they don't stop by
    themselves."""

    if any((tracker.is_alive() for tracker in self._trackers)):
      # First, gently asking the trackers to stop
      for pipe, tracker in zip(self._pipes, self._trackers):
        if tracker.is_alive():
          pipe.send(('stop', 'stop', 'stop'))
      sleep(0.1)

      # If they're not stopping, killing the trackers
      for tracker in self._trackers:
        if tracker.is_alive():
          self._log(logging.WARNING, "Tracker process did not stop properly, "
                                     "terminating it")
          tracker.terminate()

  def get_data(self,
               img: np.ndarray
               ) -> Optional[tuple[list[tuple[float, ...]], float, float]]:
    """Takes an image as an input, performs spot detection on it, computes the
    strain from the newly detected spots, and returns the spot positions and
    strain values.

    Args:
      img: The image on which the spots should be detected.

    Returns:
      A :obj:`list` containing :obj:`tuple` with the coordinates of the centers 
      of the detected spots, and the calculated x and y strain values.
    
    .. versionchanged:: 1.5.10 renamed from *get_def* to *get_data*
    """

    # Sending the latest sub-image containing the spot to track
    # Also sending the coordinates of the top left pixel
    for pipe, spot in zip(self._pipes, self.spots):
      x_top, x_bottom, y_left, y_right = spot.sorted()
      slice_y = slice(max(0, y_left - self._border),
                      min(img.shape[0], y_right + self._border))
      slice_x = slice(max(0, x_top - self._border),
                      min(img.shape[1], x_bottom + self._border))
      pipe.send((slice_y.start, slice_x.start, img[slice_y, slice_x]))

    for i, (pipe, spot) in enumerate(zip(self._pipes, self.spots)):

      # Receiving the data from the tracker, if there's any
      if pipe.poll(timeout=0.1):
        box = pipe.recv()

        # In case a tracker faced an error, stopping them all and raising
        if isinstance(box, str):
          self.stop_tracking()
          self._log(logging.ERROR, "Tracker process returned exception !")
          raise LostSpotError

        self.spots[i] = box

    overlap = False

    # Checking if the newly received boxes overlaps with each other
    for box_1, box_2 in combinations(self.spots, 2):

      if box_1 is None or box_2 is None:
        continue

      if self._overlap_box(box_1, box_2):

        # If there's overlapping in safe mode, raising directly
        if self._safe_mode:
          self.stop_tracking()
          self._log(logging.ERROR, "Overlapping detected in safe mode, "
                                   "raising exception")
          raise LostSpotError

        self._log(logging.WARNING, "Overlapping detected ! Reducing spot "
                                   "window")

        # If we're not in safe mode, simply reduce the boxes by 1 pixel
        # Also, make sure the box is not being reduced too much
        overlap = True
        x_top_1, x_bottom_1, y_left_1, y_right_1 = box_1.sorted()
        x_top_2, x_bottom_2, y_left_2, y_right_2 = box_2.sorted()

        box_1.x_start = min(x_top_1 + 1, int(box_1.x_centroid - 2))
        box_1.y_start = min(y_left_1 + 1, int(box_1.y_centroid - 2))
        box_1.x_end = max(x_bottom_1 - 1, int(box_1.x_centroid + 2))
        box_1.y_end = max(y_right_1 - 1, int(box_1.y_centroid + 2))

        box_2.x_start = min(x_top_2 + 1, int(box_2.x_centroid - 2))
        box_2.y_start = min(y_left_2 + 1, int(box_2.y_centroid - 2))
        box_2.x_end = max(x_bottom_2 - 1, int(box_2.x_centroid + 2))
        box_2.y_end = max(y_right_2 - 1, int(box_2.y_centroid + 2))

    if overlap:
      self._consecutive_overlaps += 1
      if self._consecutive_overlaps > 10:
        self._log(logging.ERROR, "Too many consecutive overlaps !")
        raise LostSpotError
    else:
      self._consecutive_overlaps = 0

    # If there are multiple spots, the x and y strains can be computed
    if len(self.spots) > 1:
      x = [spot.x_centroid for spot in self.spots if spot is not None]
      y = [spot.y_centroid for spot in self.spots if spot is not None]
      # The strain is calculated based on the positions of the extreme
      # spots in each direction
      try:
        exx = ((max(x) - min(x)) / self.spots.x_l0 - 1) * 100
      except ZeroDivisionError:
        exx = 0
      try:
        eyy = ((max(y) - min(y)) / self.spots.y_l0 - 1) * 100
      except ZeroDivisionError:
        eyy = 0
      centers = list(zip(y, x))
      return centers, eyy, exx

    # If only one spot was detected, the strain isn't computed
    else:
      x = self.spots[0].x_centroid
      y = self.spots[0].y_centroid
      return [(y, x)], 0, 0

  def _log(self, level: int, msg: str) -> None:
    """Wrapper for recording log messages.

    Also instantiates the :obj:`~logging.Logger` on the first message.

    Args:
      level: The logging level of the message, as an :obj:`int`.
      msg: The message to lof, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def _send(self,
            conn: Connection,
            val: Union[str, tuple[int, int, np.ndarray]]) -> None:
    """Wrapper for sending messages to the Tracker processes.

    In Linux, checks that the Pipe is not full before sending the message.

    Args:
      conn: The Connection to use for sending the message.
      val: The message to send to the Tracker process.
    """

    if self._system == 'Linux':
      if select([], [conn], [], 0)[1]:
        conn.send(val)
      elif time() - self._last_warn > 1:
          self._last_warn = time()
          self._log(logging.WARNING, f"Cannot send the image to process to the"
                                     f" Tracker process, the Pipe is full !")
    else:
      conn.send(val)

  @staticmethod
  def _overlap_box(box_1: Box, box_2: Box) -> bool:
    """Determines whether two boxes are overlapping or not."""

    x_min_1, x_max_1, y_min_1, y_max_1 = box_1.sorted()
    x_min_2, x_max_2, y_min_2, y_max_2 = box_2.sorted()

    return max((min(x_max_1, x_max_2) - max(x_min_1, x_min_2)), 0) * max(
      (min(y_max_1, y_max_2) - max(y_min_1, y_min_2)), 0) > 0

  @staticmethod
  def _overlap_bbox(prop_1, prop_2) -> bool:
    """Determines whether two bboxes are overlapping or not."""

    y_min_1, x_min_1, y_max_1, x_max_1 = prop_1.bbox
    y_min_2, x_min_2, y_max_2, x_max_2 = prop_2.bbox

    return max((min(x_max_1, x_max_2) - max(x_min_1, x_min_2)), 0) * max(
      (min(y_max_1, y_max_2) - max(y_min_1, y_min_2)), 0) > 0

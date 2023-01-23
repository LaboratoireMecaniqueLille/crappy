# coding: utf-8

from multiprocessing import Pipe, current_process
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from typing import Optional, Tuple, List, Union
import numpy as np
from itertools import combinations
from time import sleep, time
import logging
import logging.handlers
from select import select

from ...camera_config import SpotsBoxes, Box
from .tracker import Tracker, LostSpotError


class VideoExtensoTool:
  """This class tracks up to 4 spots on successive images, and computes the
  strain on the material based on the displacement of the spots from their
  original position.

  The first step is to detect the spots to track. Once this is done, the
  initial distances in x and y between the spots is saved. For each spot, a
  Process in charge of tracking it is started. When a new image is received,
  subframes based on the last known positions of the spots are sent to the
  tracker processes, and they return the new positions of the detected spots.
  Based on these new positions, updated strain values are returned.

  It is possible to track only one spot, in which case only the position of its
  center is returned and the strain values are left to 0.
  """

  def __init__(self,
               spots: SpotsBoxes,
               x_l0: float,
               y_l0: float,
               thresh: int,
               log_level: Optional[int],
               log_queue: Queue,
               white_spots: bool = False,
               update_thresh: bool = False,
               num_spots: Optional[int] = None,
               safe_mode: bool = False,
               border: int = 5,
               blur: Optional[int] = 5) -> None:
    """Sets the args and the other instance attributes.

    Args:
      white_spots: If :obj:`True`, detects white objects on a black background,
        else black objects on a white background.
      update_thresh: If :obj:`True`, the threshold for detecting the spots is
        re-calculated for each new image. Otherwise, the first calculated
        threshold is kept for the entire test. The spots are less likely to be
        lost with adaptive threshold, but the measurement will be more noisy.
        Adaptive threshold may also yield inconsistent results when spots are
        lost.
      num_spots: The number of spots to detect, between 1 and 4. The class will
        then try to detect this exact number of spots, and won't work if not
        enough spots can be found. If this argument is not given, at most 4
        spots can be detected but the class will work with any number of
        detected spots between 1 and 4.
      safe_mode: If :obj:`True`, the class will stop and raise an exception as
        soon as overlapping is detected. Otherwise, it will first try to reduce
        the detection window to get rid of overlapping. This argument should be
        used when inconsistency in the results may have critical consequences.
      border: When searching for the new position of a spot, the class will
        search in the last known bounding box of this spot plus a few
        additional pixels in each direction. This argument sets the number of
        additional pixels to use. It should be greater than the expected
        "speed" of the spots, in pixels / frame. But if it's too big, noise or
        other spots might hinder the detection.
      blur: The size in pixels of the kernel to use for applying a median blur
        to the image before the spot detection. If not given, no blurring is
        performed. A slight blur improves the spot detection by smoothening the
        noise, but also takes a bit more time compared to no blurring.
    """

    # These attributes will be used later
    self._consecutive_overlaps = 0
    self._trackers = list()
    self._pipes = list()

    if num_spots is not None and num_spots not in range(1, 5):
      raise ValueError("num_spots should be either None, 1, 2, 3 or 4 !")
    self._num_spots = num_spots

    # Setting the args
    self._white_spots = white_spots
    self._update_thresh = update_thresh
    self._safe_mode = safe_mode
    self._border = border
    self._blur = blur
    self.spots = spots
    self._x_l0 = x_l0
    self._y_l0 = y_l0
    self._thresh = thresh

    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._log_queue = log_queue

    self._last_warn = time()

  def __del__(self) -> None:
    """Security to ensure there are no zombie processes left when exiting."""

    self.stop_tracking()

  def start_tracking(self) -> None:
    """Creates a Tracker process for each detected spot, and starts it.

    Also creates a Pipe for each spot to communicate with the Tracker process.
    """

    if self.spots.empty():
      raise AttributeError("[VideoExtenso] No spots selected, aborting !")

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
    """Stops all the active Tracker processes, either gently or by terminating
    them if they don't stop by themselves."""

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
               img: np.ndarray) -> Optional[Tuple[List[Tuple[float, ...]],
                                                  float, float]]:
    """Takes an image as an input, performs spot detection on it, computes the
    strain from the newly detected spots, and returns the spot positions and
    strain values.

    Args:
      img: The image on which the spots should be detected.

    Returns:
      A :obj:`list` containing tuples with the coordinates of the centers of
      the detected spots, and the calculated x and y strain values.
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
        box_1.x_start = min(x_top_1 + 1, box_1.x_centroid - 2)
        box_1.y_start = min(y_left_1 + 1, box_1.y_centroid - 2)
        box_1.x_end = max(x_bottom_1 - 1, box_1.x_centroid + 2)
        box_1.y_end = max(y_right_1 - 1, box_1.y_centroid + 2)
        box_2.x_start = min(x_top_2 + 1, box_2.x_centroid - 2)
        box_2.y_start = min(y_left_2 + 1, box_2.y_centroid - 2)
        box_2.x_end = max(x_bottom_2 - 1, box_2.x_centroid + 2)
        box_2.y_end = max(y_right_2 - 1, box_2.y_centroid + 2)

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
      try:
        # The strain is calculated based on the positions of the extreme
        # spots in each direction
        exx = ((max(x) - min(x)) / self._x_l0 - 1) * 100
        eyy = ((max(y) - min(y)) / self._y_l0 - 1) * 100
      except ZeroDivisionError:
        # Shouldn't happen but adding a safety just in case
        exx, eyy = 0, 0
      centers = list(zip(y, x))
      return centers, eyy, exx

    # If only one spot was detected, the strain isn't computed
    else:
      x = self.spots[0].x_centroid
      y = self.spots[0].y_centroid
      return [(y, x)], 0, 0

  def _log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def _send(self,
            conn: Connection,
            val: Union[str, Tuple[int, int, np.ndarray]]) -> None:
    """"""

    if select([], [conn], [], 0)[1]:
      conn.send(val)
    else:
      if time() - self._last_warn > 1:
        self._last_warn = time()
        self._log(logging.WARNING, f"Cannot send the image to process to the "
                                   f"Tracker process, the Pipe is full !")

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

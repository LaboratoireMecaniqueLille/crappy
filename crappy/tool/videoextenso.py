# coding: utf-8

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Optional, Tuple, List
import numpy as np
from itertools import combinations
from time import sleep
from warnings import warn
from .._global import OptionalModule
from .cameraConfigBoxes import Spot_boxes, Box

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
try:
  from skimage.filters import threshold_otsu
  from skimage.morphology import label
  from skimage.measure import regionprops
except (ModuleNotFoundError, ImportError):
  label = OptionalModule("skimage", "Please install scikit-image to use"
                         "Video-extenso")
  threshold_otsu = regionprops = label


class LostSpotError(Exception):
  """Exception raised when a spot is lost, or when there's too much
  overlapping."""

  pass


class VideoExtenso:
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
               white_spots: bool = False,
               update_thresh: bool = False,
               num_spots: Optional[int] = None,
               safe_mode: bool = False,
               border: int = 5,
               min_area: int = 150,
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
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of camera and
        the size of the spots to detect.
      blur: The size in pixels of the kernel to use for applying a median blur
        to the image before the spot detection. If not given, no blurring is
        performed. A slight blur improves the spot detection by smoothening the
        noise, but also takes a bit more time compared to no blurring.
    """

    warn("The VideoExtenso class will be renamed to VideoExtensoTool in "
         "version 2.0.0", DeprecationWarning)
    warn("The num_spots argument will be removed in version 2.0.0",
         DeprecationWarning)
    warn("The min_area argument will be removed in version 2.0.0",
         DeprecationWarning)

    if num_spots is not None and num_spots not in range(1, 5):
      raise ValueError("num_spots should be either None, 1, 2, 3 or 4 !")
    self._num_spots = num_spots

    # Setting the args
    self._white_spots = white_spots
    self._update_thresh = update_thresh
    self._safe_mode = safe_mode
    self._border = border
    self._min_area = min_area
    self._blur = blur

    # Setting the other attributes
    self._consecutive_overlaps = 0
    self.spots = Spot_boxes()
    self._thresh = 0 if white_spots else 255
    self.x_l0 = None
    self.y_l0 = None
    self._trackers = list()
    self._pipes = list()

  def __del__(self) -> None:
    """Security to ensure there are no zombie processes left when exiting."""

    self.stop_tracking()

  def detect_spots(self,
                   img: np.ndarray,
                   y_orig: int,
                   x_orig: int) -> Optional[Spot_boxes]:
    """Transforms the image to improve spot detection, detects up to 4 spots
    and return a Spot_boxes object containing all the detected spots.

    Args:
      img: The sub-image on which the spots should be detected.
      y_orig: The y coordinate of the top-left pixel of the sub-image on the
        entire image.
      x_orig: The x coordinate of the top-left pixel of the sub-image on the
        entire image.

    Returns:
      A Spot_boxes object containing all the detected spots.
    """

    warn("The detect_spots method will be removed in version 2.0.0",
         DeprecationWarning)

    # First, blurring the image if asked to
    if self._blur is not None and self._blur > 1:
      img = cv2.medianBlur(img, self._blur)

    # Determining the best threshold for the image
    self._thresh = threshold_otsu(img)

    # Getting all pixels superior or inferior to threshold
    if self._white_spots:
      black_white = img > self._thresh
    else:
      black_white = img <= self._thresh

    # The original image is not needed anymore
    del img

    # Detecting the spots on the image
    props = regionprops(label(black_white))

    # The black and white image is not needed anymore
    del black_white

    # Removing regions that are too small or not circular
    props = [prop for prop in props if prop.solidity > 0.8
             and prop.area > self._min_area]

    # Detecting overlapping spots and storing the ones that should be removed
    to_remove = list()
    for prop_1, prop_2 in combinations(props, 2):
      if self._overlap_bbox(prop_1, prop_2):
        to_remove.append(prop_2 if prop_1.area > prop_2.area else prop_1)

    # Now removing the overlapping spots
    if to_remove:
      print("[VideoExtenso] Overlapping spots found, removing the smaller "
            "overlapping ones")
    for prop in to_remove:
      props.remove(prop)

    # Sorting the regions by area
    props = sorted(props, key=lambda prop: prop.area, reverse=True)

    # Keeping only the biggest areas to match the target number of spots
    if self._num_spots is not None:
      props = props[:self._num_spots]
    else:
      props = props[:4]

    # Indicating the user if not enough spots were found
    if not props:
      print("[VideoExtenso] No spots found !")
      return
    elif self._num_spots is not None and len(props) != self._num_spots:
      print(f"[VideoExtenso] Expected {self._num_spots} spots, "
            f"found only {len(props)}")
      return

    # Replacing the previously detected spots with the new ones
    self.spots.reset()
    for i, prop in enumerate(props):
      # Extracting the properties of interest
      y, x = prop.centroid
      y_min, x_min, y_max, x_max = prop.bbox
      # Adjusting with the offset given as argument
      x += x_orig
      x_min += x_orig
      x_max += x_orig
      y += y_orig
      y_min += y_orig
      y_max += y_orig

      self.spots[i] = Box(x_start=x_min, x_end=x_max,
                          y_start=y_min, y_end=y_max,
                          x_centroid=x, y_centroid=y)

    return self.spots

  def save_length(self) -> bool:
    """Saves the initial length in x and y between the detected spots."""

    warn("The save_length method will be removed in version 2.0.0",
         DeprecationWarning)

    # Cannot determine a length if no spots was detected
    if self.spots.empty():
      print("[VideoExtenso] Cannot save L0, no spots selected yet !")
      return False

    # Simply taking the distance between the extrema as the initial length
    if len(self.spots) > 1:
      x_centers = [spot.x_centroid for spot in self.spots if spot is not None]
      y_centers = [spot.y_centroid for spot in self.spots if spot is not None]
      self.x_l0 = max(x_centers) - min(x_centers)
      self.y_l0 = max(y_centers) - min(y_centers)

    # If only one spot detected, setting the initial lengths to 0
    else:
      self.x_l0 = 0
      self.y_l0 = 0

    return True

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
      sleep(0.05)

      # If they're not stopping, killing the trackers
      for tracker in self._trackers:
        if tracker.is_alive():
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

    # First, saving the initial length if not already saved
    if self.x_l0 is None or self.y_l0 is None:
       if not self.save_length():
         # If no spots are selected, it should be detected here
         raise IOError("[VideoExtenso] No spots selected, aborting !")

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
      if pipe.poll(timeout=0.5):
        box = pipe.recv()

        # In case a tracker faced an error, stopping them all and raising
        if isinstance(box, str):
          self.stop_tracking()
          raise LostSpotError(f"[VideoExtenso] Tracker returned error : {box}")

        self.spots[i] = box

    overlap = False

    # Checking if the newly received boxes overlaps with each other
    for box_1, box_2 in combinations(self.spots, 2):
      if self._overlap_box(box_1, box_2):

        # If there's overlapping in safe mode, raising directly
        if self._safe_mode:
          self.stop_tracking()
          raise LostSpotError("[VideoExtenso] Overlapping detected in safe"
                              " mode, raising exception directly")

        print("[VideoExtenso] Overlapping detected ! Reducing spot window")

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
        raise LostSpotError("[VideoExtenso] Too many consecutive "
                            "overlaps !")
    else:
      self._consecutive_overlaps = 0

    # If there are multiple spots, the x and y strains can be computed
    if len(self.spots) > 1:
      x = [spot.x_centroid for spot in self.spots if spot is not None]
      y = [spot.y_centroid for spot in self.spots if spot is not None]
      try:
        # The strain is calculated based on the positions of the extreme
        # spots in each direction
        exx = ((max(x) - min(x)) / self.x_l0 - 1) * 100
        eyy = ((max(y) - min(y)) / self.y_l0 - 1) * 100
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


class Tracker(Process):
  """Process whose task is to track a spot on an image.

  It receives a subframe centered on the last known position of the spot, and
  returns the updated position of the detected spot.
  """

  def __init__(self,
               pipe: Connection,
               white_spots: bool = False,
               thresh: Optional[int] = None,
               blur: Optional[float] = 5) -> None:
    """Sets the args.

    Args:
      pipe: The :obj:`multiprocessing.connection.Connection` object through
        which the image is received and the updated coordinates of the spot
        are sent back.
      white_spots: If :obj:`True`, detects white objects on a black background,
        else black objects on a white background.
      thresh: If given, this threshold value will always be used for isolating
        the spot from the background. If not given (:obj:`None`), a new
        threshold is recalculated for each new subframe. Spots are less likely
        to be lost with an adaptive threshold, but it takes a bit more time.
      blur: If not :obj:`None`, the subframe is first blurred before trying to
        detect the spot. This argument gives the size of the kernel to use for
        blurring. Better results are obtained with blurring, but it takes a bit
        more time.
    """

    super().__init__()

    self._pipe = pipe
    self._white_spots = white_spots
    self._thresh = thresh
    self._blur = blur

    self._n = 0

  def run(self) -> None:
    """Continuously reads incoming subframes, tries to detect a spot and sends
    back the coordinates of the detected spot.

    Can only be stopped either with a :exc:`KeyboardInterrupt` or when
    receiving a text message from the parent VideoExtenso class.
    """

    # Looping forever for receiving data
    try:
      while True:
        # Making sure the call to recv is not blocking
        if self._pipe.poll(0.5):
          y_start, x_start, img = self._pipe.recv()
          self._n += 1

          # If a string is received, always means the process has to stop
          if isinstance(img, str):
            break

          # Simply sending back the new Box containing the spot
          try:
            self._pipe.send(self._evaluate(x_start, y_start, img))

          # If the caught exception is a KeyboardInterrupt, simply stopping
          except KeyboardInterrupt:
            break
          # Sending back the exception if anything else unexpected happened
          except (Exception,) as exc:
            self._pipe.send(str(exc))

    # In case the user presses CTRL+C, simply stopping the process
    except KeyboardInterrupt:
      pass

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
      A Box object containing the x and y start and end positions of the
      detected spot, as well as the coordinates of the centroid.
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
        print("[VideoExtenso Tracker] Detected spot too small compared with "
              "overall box size, recalculating threshold")
        thresh = threshold_otsu(img)
        if self._white_spots:
          black_white = (img > thresh).astype('uint8')
        else:
          black_white = (img <= thresh).astype('uint8')

        # If the spot still cannot be detected, aborting
        if np.count_nonzero(black_white) < 0.1 * img.size:
          raise LostSpotError("Couldn't detect spot with adaptive threshold, "
                              "aborting !")

      # If an adaptive threshold is already used, nothing more can be done
      else:
        raise LostSpotError("Couldn't detect spot with adaptive threshold, "
                            "aborting !")

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

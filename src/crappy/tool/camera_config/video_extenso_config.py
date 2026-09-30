# coding: utf-8

"""Tk-backed VideoExtenso configuration."""

from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config_boxes import CameraConfigBoxes
from .config_core import VideoExtensoBehavior
from .config_tools import SpotsDetector, SpotsBoxes
from ...camera.meta_camera import Camera


class VideoExtensoConfig(VideoExtensoBehavior, CameraConfigBoxes):
  """Configure initial spot detection for a VideoExtenso Block.

  Drag a box over the source image to detect spots within that crop. The
  window owns a :class:`~crappy.tool.camera_config.config_tools.SpotsDetector`
  and exposes a Save L0 action for their initial separation. On close,
  :meth:`get_config` exports the spot boxes and threshold, not the detector.
  Detection, validation, and Save L0 behavior live in
  :class:`~crappy.tool.camera_config.config_core.selection_behavior.\
VideoExtensoBehavior`.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *VE_config* to *VideoExtensoConfig*
  .. versionchanged:: 2.1.0 creates and owns its SpotsDetector, and exports
     only the initial detection result
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               white_spots: bool,
               num_spots: int | None,
               min_area: int,
               blur: int | None,
               update_thresh: bool,
               safe_mode: bool,
               border: int) -> None:
    """Initialize the preview and detector with the requested spot policy.

    Args:
      camera: The :class:`~crappy.camera.meta_camera.camera.Camera` object in
        charge of acquiring the images.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      max_freq: The maximum frequency this window is allowed to loop at. It is
        simply the ``freq`` attribute of the :class:`~crappy.blocks.Camera`
        Block.

        .. versionadded:: 2.0.0
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output.

        .. versionadded:: 2.1.0
      white_spots: If :obj:`True`, detects white spots over a black background.
        If :obj:`False`, detects black spots over a white background.
      num_spots: The number of spots to detect, as an :obj:`int` between `1`
        and `4`. If given, will try to detect exactly that number of spots and
        will fail if not enough spots can be detected. If left to :obj:`None`,
        will detect up to `4` spots, but potentially fewer.
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of the camera
        and the size of the spots to detect.
      blur: An :obj:`int`, odd and greater than `1`, defining the size of the
        kernel to use when applying a median blur filter to the image before
        trying to detect spots. Can also be set to :obj:`None`, in which case
        no median blur filter is applied before detecting the spots.
      update_thresh: If :obj:`True`, the gray level threshold for detecting
        the spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost. This setting is not used during initial detection; the
        public VideoExtenso Block independently supplies the same setting to
        the runtime processing layer.
      safe_mode: If :obj:`True`, will stop and raise an exception as soon as
        overlapping spots are detected. Otherwise, will first try to reduce the
        detection window to get rid of overlapping. This argument should be
        used when inconsistency in the results may have critical consequences.
        This setting is not used during initial detection; the public
        VideoExtenso Block independently supplies the same setting to the
        runtime processing layer.
      border: When searching for the new position of a spot, will search in the
        last known bounding box of this spot plus a few additional pixels in
        each direction. This argument sets the number of additional pixels to
        use. It should be greater than the expected "speed" of the spots, in
        pixels / frame. But if set too high, noise or other spots might hinder
        the detection. This setting is not used during initial detection; the
        public VideoExtenso Block independently supplies the same setting to
        the runtime processing layer.

    .. versionchanged:: 1.5.10 renamed *ve* argument to *video_extenso*
    .. versionremoved:: 2.0.0 *video_extenso* argument
    """

    super().__init__(camera, log_queue, log_level, max_freq, transform)
    self._detector: SpotsDetector = SpotsDetector(white_spots=white_spots,
                                                  num_spots=num_spots,
                                                  min_area=min_area,
                                                  blur=blur,
                                                  update_thresh=update_thresh,
                                                  safe_mode=safe_mode,
                                                  border=border)
    self._spots: SpotsBoxes = self._detector.spots

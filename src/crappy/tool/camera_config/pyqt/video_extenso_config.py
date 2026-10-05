# coding: utf-8

"""PyQt6 camera configurator for :class:`~crappy.blocks.VideoExtenso`."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import VideoExtensoConfig
from ..config_tools import SpotsDetector
from ....camera.meta_camera import Camera
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtVideoExtensoConfig(VideoExtensoConfig, PyQtCameraConfigBoxes):
  """PyQt6 window for initial video-extensometry spot detection.

  A left-button drag detects spots in the selected crop of the image. The
  window owns a :class:`~crappy.tool.camera_config.config_tools.SpotsDetector`
  and provides a Save L0 action for recording initial horizontal and vertical
  distance. A valid close saves those lengths if unset.
  :meth:`get_config() <crappy.tool.camera_config.base.video_extenso_config.\
VideoExtensoConfig.get_config>` returns the selected spot boxes and the
  computed threshold. Detection and validation rules are inherited from the
  shared :class:`~crappy.tool.camera_config.base.video_extenso_config.\
VideoExtensoConfig`.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
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
    """Builds the window and its spot detector.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.
      white_spots: :obj:`True` for light spots on a dark background,
        :obj:`False` for dark spots on a light background.
      num_spots: Required number of spots from 1 to 4, or :obj:`None` to accept
        up to 4. An unsuccessful detection leaves the previous selection
        unchanged.
      min_area: Area in pixels that a detected region must exceed.
      blur: Median-filter kernel size, normally an odd integer greater than 1.
        :obj:`None` disables filtering.
      update_thresh: Runtime tracking policy forwarded for compatibility. It is
        not used during initial detection.
      safe_mode: Runtime overlap policy forwarded for compatibility. It is not
        used during initial detection.
      border: Runtime tracking margin forwarded for compatibility. It is not
        used during initial detection.
    """

    super().__init__(camera, log_queue, log_level, max_freq, transform)
    self._detector = SpotsDetector(white_spots=white_spots,
                                   num_spots=num_spots,
                                   min_area=min_area,
                                   blur=blur,
                                   update_thresh=update_thresh,
                                   safe_mode=safe_mode,
                                   border=border)
    self._spots = self._detector.spots

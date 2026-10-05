# coding: utf-8

"""PyQt6 camera configurator for :class:`~crappy.blocks.DISCorrel`."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import DISCorrelConfig
from ..config_tools import Box
from ....camera.meta_camera import Camera
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtDISCorrelConfig(DISCorrelConfig, PyQtCameraConfigBoxes):
  """PyQt6 window for selecting a :class:`~crappy.blocks.DISCorrel` region of
  interest (ROI).

  A left-button drag replaces the provided correlation
  :class:`~crappy.tool.camera_config.config_tools.Box` when a nonempty
  selection is released. Closing requires a selected ROI. Selection and
  validation rules are inherited from the shared
  :class:`~crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig`.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patch: Box) -> None:
    """Builds the window with the provided correlation region.

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
      patch: :class:`~crappy.tool.camera_config.config_tools.Box` to display
        and update in place. The same
        :class:`~crappy.tool.camera_config.config_tools.Box` is returned by
        :meth:`get_config() <crappy.tool.camera_config.base.dis_correl_config.\
DISCorrelConfig.get_config>` after validation.
    """

    # The overlay must exist before the first image is drawn
    self._correl_box = patch
    self._draw_correl_box = True
    super().__init__(camera, log_queue, log_level, max_freq, transform)

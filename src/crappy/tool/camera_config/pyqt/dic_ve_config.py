# coding: utf-8

"""PyQt6 camera configurator for :class:`~crappy.blocks.DICVE`."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import DICVEConfig
from ..config_tools import SpotsBoxes
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import CameraScaleSetting
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtDICVEConfig(DICVEConfig, PyQtCameraConfigBoxes):
  """PyQt6 window for selecting digital image correlation tracking patches.

  Dragging a rectangle large enough for the Patch size setting positions four
  patches around its edges. Apply Settings or Auto apply controls the local
  patch size. The supplied
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` collection is
  updated in place. A valid close saves initial horizontal and vertical patch
  distance. Selection rules are inherited from the shared
  :class:`~crappy.tool.camera_config.base.dic_ve_config.DICVEConfig`.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patches: SpotsBoxes) -> None:
    """Builds the window with the caller-owned tracking-patch collection.

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
      patches: :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes`
        collection to display and update in place. The same collection is
        returned by
        :meth:`get_config() <crappy.tool.camera_config.base.dic_ve_config.\
DICVEConfig.get_config>` after validation.
    """

    self._patch_size: CameraScaleSetting | None = None
    super().__init__(camera, log_queue, log_level, max_freq, transform)
    self._spots = patches

# coding: utf-8

"""Tk-backed DICVE configuration."""

from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config_boxes import CameraConfigBoxes
from .config_core import DICVEBehavior
from .config_tools import SpotsBoxes
from ...camera.meta_camera import Camera
from ...camera.meta_camera.camera_setting import CameraScaleSetting


class DICVEConfig(DICVEBehavior, CameraConfigBoxes):
  """Configure four tracking patches for a DICVE Block.

  Dragging a sufficiently large selection positions four patches around its
  edges. Their size comes from the local Patch size setting. The supplied
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` is updated in
  place, and its initial separation is saved after valid close. Patch layout
  and validation live in :class:`~crappy.tool.camera_config.config_core.\
selection_behavior.DICVEBehavior`.

  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *DISVE_config* to *DICVEConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patches: SpotsBoxes) -> None:
    """Initialize the configurator with the caller-owned patch collection.

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
      patches: An instance of
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` containing
        the patches to follow for image correlation.
    """

    self._patch_size: CameraScaleSetting | None = None

    super().__init__(camera, log_queue, log_level, max_freq, transform)

    self._spots: SpotsBoxes = patches

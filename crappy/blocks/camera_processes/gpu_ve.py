# coding: utf-8

from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.image_processing import GpuCorrelTool
from ...tool.camera_config import SpotsBoxes
from ..._global import OptionalModule

try:
  import pycuda.tools
  import pycuda.driver
except (ModuleNotFoundError, ImportError):
  pycuda = OptionalModule("pycuda")


class GpuVeProcess(CameraProcess):
  """"""

  def __init__(self,
               patches: List[Tuple[int, int, int, int]],
               log_queue: Queue,
               log_level: int = 20,
               verbose: int = 0,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               img_ref: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     verbose=bool(verbose))

    pycuda.driver.init()
    context = pycuda.tools.make_default_context()

    self._gpuve_kw = dict(context=context,
                          verbose=verbose,
                          levels=1,
                          resampling_factor=2,
                          kernel_file=kernel_file,
                          iterations=iterations,
                          fields=['x', 'y'],
                          ref_img=img_ref,
                          mask=None,
                          mul=mul)

    self._correls: Optional[List[GpuCorrelTool]] = None
    self._patches = patches
    self._img_ref = img_ref

    self._spots = SpotsBoxes()
    self._spots.set_spots(patches)

    self._img0_set = img_ref is not None

  def _init(self) -> None:
    """"""

    self._log(logging.INFO, "Instantiating the GPUCorrel tool instances")
    self._gpuve_kw.update(logger_name=self.name)
    self._correls = [GpuCorrelTool(**self._gpuve_kw) for _ in self._patches]

    # We can already set the sizes of the images as they are already known
    self._log(logging.INFO, "Setting the sizes of the patches")
    for correl, (_, __, h, w) in zip(self._correls, self._patches):
      correl.set_img_size((h, w))

    if self._img_ref is not None:
      self._log(logging.INFO, "Initializing the GPUCorrel tool instances "
                              "with the given reference image and preparing "
                              "them")
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(
          self._img_ref[oy:oy + h, ox:ox + w].astype(np.float32))
        correl.prepare()

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return
    self.fps_count += 1

    if not self._img0_set:
      self._log(logging.INFO, "Setting the reference image")
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(self._img[oy:oy + h,
                        ox:ox + w].astype(np.float32))
        correl.prepare()
      self._img0_set = True
      return

    self._log(logging.DEBUG, "Processing the received image")
    data = [self._metadata['t(s)'], self._metadata]
    for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
      data.extend(correl.get_disp(
        self._img[oy:oy + h, ox:ox + w].astype(np.float32)).tolist())
    self._send(data)

    self._send_box(self._spots)

  def _finish(self) -> None:
    """"""

    if self._correls is not None:
      self._log(logging.INFO, "Cleaning up the GPUCorrel instances")
      for correl in self._correls:
        correl.clean()

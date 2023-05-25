# coding: utf-8

from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.image_processing import GPUCorrelTool
from ...tool.camera_config import SpotsBoxes
from ..._global import OptionalModule

try:
  import pycuda.tools
  import pycuda.driver
except (ModuleNotFoundError, ImportError):
  pycuda = OptionalModule("pycuda")


class GPUVEProcess(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can perform
  GPU-accelerated video-extensometry by tracking patches using digital image
  correlation. It then returns the position of the tracked patches.

  It is used by the :class:`~crappy.blocks.GPUVE` Block to parallelize the
  image processing and the image acquisition. It delegates most of the
  computation to the :class:`~crappy.tool.image_processing.GPUCorrelTool`. It
  is from this class that the output values are sent to the downstream Blocks,
  and that the :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` are
  sent to the :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess
  for display.
  """

  def __init__(self,
               patches: List[Tuple[int, int, int, int]],
               log_queue: Queue,
               log_level: int = 20,
               verbose: int = 0,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               img_ref: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      patches: A :obj:`list` containing the coordinates of the patches to
        track, as a :obj:`tuple` for each patch. Each tuple should contain
        exactly `4` elements, giving in pixels the `y` origin, `x` origin,
        height and width of the patch.
      log_queue: A :obj:`~multiprocessing.Queue` for sending the log messages
        to the main :obj:`~logging.Logger`, only used in Windows.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is displayed, and at level `3` so much information
        is displayed that is slows the code down. This argument is passed to
        the :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used
        in this class.
      kernel_file: The path to the file containing the kernels to use for the
        correlation. Can be a :obj:`pathlib.Path` object or a :obj:`str`. If
        not provided, the default :ref:`GPU Kernels` are used. This argument is
        passed to the :class:`~crappy.tool.image_processing.GPUCorrelTool` and
        not used in this class.
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing. This argument is passed to the
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.
      img_ref: A reference image for the correlation, given as a 2D
        :obj:`numpy.array` with `dtype` :obj:`numpy.float32`. If not given, the
        first acquired frame will be used as the reference image instead. This
        argument is passed to the
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk to go past the solution and to diverge. If it's too
        low, the convergence will be slower and require more iterations. `3`
        was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast. This argument is passed
        to the :class:`~crappy.tool.image_processing.GPUCorrelTool` and not
        used in this class.
    """

    super().__init__(log_queue=log_queue, log_level=log_level,
                     display_freq=bool(verbose))

    # Making a CUDA context common to all the patches
    pycuda.driver.init()
    context = pycuda.tools.make_default_context()

    self._gpu_ve_kw = dict(context=context,
                           verbose=verbose,
                           levels=1,
                           resampling_factor=2,
                           kernel_file=kernel_file,
                           iterations=iterations,
                           fields=['x', 'y'],
                           ref_img=img_ref,
                           mask=None,
                           mul=mul)

    self._correls: Optional[List[GPUCorrelTool]] = None
    self._patches = patches
    self._img_ref = img_ref

    self._spots = SpotsBoxes()
    self._spots.set_spots(patches)

    self._img0_set = img_ref is not None

  def _init(self) -> None:
    """Initializes the GPUCorrelTool instances, and set their reference image
    if a ``img_ref`` argument was provided."""

    # Instantiating the GPUCorrelTool instances
    self._log(logging.INFO, "Instantiating the GPUCorrel tool instances")
    self._gpu_ve_kw.update(logger_name=self.name)
    self._correls = [GPUCorrelTool(**self._gpu_ve_kw) for _ in self._patches]

    # We can already set the sizes of the images as they are already known
    self._log(logging.INFO, "Setting the sizes of the patches")
    for correl, (_, __, h, w) in zip(self._correls, self._patches):
      correl.set_img_size((h, w))

    # Setting the reference image if it was given as an argument
    if self._img_ref is not None:
      self._log(logging.INFO, "Initializing the GPUCorrel tool instances "
                              "with the given reference image and preparing "
                              "them")
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(
          self._img_ref[oy:oy + h, ox:ox + w].astype(np.float32))
        correl.prepare()

  def _loop(self) -> None:
    """This method grabs the latest frame and gives it for processing to the
    several instances of :class:`~crappy.tool.image_processing.GPUCorrelTool`.
    Then sends the displacement data to the downstream Blocks.

    If there's no new frame grabbed, doesn't do anything. On the first acquired
    frame, does not process it but initializes the instances of GPUCorrelTool
    with it instead if no reference image was given as argument. Also sends the
    patches for display to the
    :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess.
    """

    # Nothing to do if no new frame was grabbed
    if not self._get_data():
      return

    self.fps_count += 1

    # Setting the reference image with the first received frame if it was not
    # given as an argument
    if not self._img0_set:
      self._log(logging.INFO, "Setting the reference image")
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(self._img[oy:oy + h,
                        ox:ox + w].astype(np.float32))
        correl.prepare()
      self._img0_set = True
      return

    # Performing the image correlation
    self._log(logging.DEBUG, "Processing the received image")
    data = [self._metadata['t(s)'], self._metadata]
    for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
      data.extend(correl.get_disp(
        self._img[oy:oy + h, ox:ox + w].astype(np.float32)).tolist())

    # Sending the data to downstream Blocks
    self._send(data)

    # Sending the patches to the Displayer for display
    self._send_box(self._spots)

  def _finish(self) -> None:
    """Performs cleanup on the several
    :class:`~crappy.tool.image_processing.GPUCorrelTool` used."""

    if self._correls is not None:
      self._log(logging.INFO, "Cleaning up the GPUCorrel instances")
      for correl in self._correls:
        correl.clean()

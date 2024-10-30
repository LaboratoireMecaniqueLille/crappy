# coding: utf-8

from typing import Optional, Literal
import numpy as np
import logging
import logging.handlers
from time import sleep

from .camera_process import CameraProcess
from ...tool.image_processing import DICVETool
from ...tool.camera_config import SpotsBoxes


class DICVEProcess(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can perform
  video-extensometry by tracking patches on images using various Digital Image 
  Correlation techniques.
  
  It is used by the :class:`~crappy.blocks.DICVE` Block to parallelize the
  image processing and the image acquisition. It delegates most of the 
  computation to the :class:`~crappy.tool.image_processing.DICVETool`. It is
  from this class that the output values are sent to the downstream Blocks, and
  that the :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` are sent
  to the :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess for
  display.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self,
               patches: SpotsBoxes,
               method: Literal['Disflow', 'Lucas Kanade',
                               'Pixel precision', 'Parabola'] = 'Disflow',
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               border: float = 0.2,
               safe: bool = True,
               follow: bool = True,
               raise_on_exit: bool = True) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      patches: An instance of the
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` class,
        containing the coordinates of the patches to track. This argument is
        passed to the :obj:`~crappy.tool.image_processing.DICVETool` and not
        used in this class.
      method: The method to use to calculate the displacement. `Disflow` uses
        opencv's DISOpticalFlow and `Lucas Kanade` uses opencv's
        calcOpticalFlowPyrLK, while all other methods are based on a basic
        cross-correlation in the Fourier domain. `Pixel precision` calculates
        the displacement by getting the position of the maximum of the
        cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. `Parabola` refines the result of
        `Pixel precision` by interpolating the neighborhood of the maximum, and
        has thus a sub-pixel resolution. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      alpha: Weight of the smoothness term in DISFlow, as a :obj:`float`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      delta: Weight of the color constancy term in DISFlow, as a :obj:`float`.
        This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      gamma: Weight of the gradient constancy term in DISFlow , as a
        :obj:`float`. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DISFlow (`0` means full scale), as an :obj:`int`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DISFlow, as an :obj:`int`. This argument
        is passed to the :obj:`~crappy.tool.image_processing.DICVETool` and not
        used in this class.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DISFlow, as an :obj:`int`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      patch_size: Size of an image patch for matching in DISFlow
        (in pixels). This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      patch_stride: Stride between neighbor patches in DISFlow. Must be
        less than patch size. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      border: Crop the patch on each side according to this value before
        calculating the displacements. 0 means no cropping, 1 means the entire
        patch is cropped. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      safe: If :obj:`True`, checks whether the patches aren't exiting the
        image, and raises an error if that's the case. This argument is passed
        to the :obj:`~crappy.tool.image_processing.DICVETool` and not used in
        this class.
      follow: It :obj:`True`, the patches will move to follow the displacement
        of the image. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DICVETool` and not used in this
        class.
      raise_on_exit: If :obj:`True`, raises an exception and stops the test
        when losing the patches. Otherwise, simply stops processing but lets 
        the test go on.
    """

    super().__init__()

    # Arguments to pass to the DICVETool
    self._patches = patches
    self._method = method
    self._alpha = alpha
    self._delta = delta
    self._gamma = gamma
    self._finest_scale = finest_scale
    self._iterations = iterations
    self._gradient_iterations = gradient_iterations
    self._patch_size = patch_size
    self._patch_stride = patch_stride
    self._border = border
    self._safe = safe
    self._follow = follow
    
    # Other attributes
    self._raise_on_exit = raise_on_exit
    self._disve: Optional[DICVETool] = None
    self._img0_set = False
    self._lost_patch = False

  def init(self) -> None:
    """Instantiates the :obj:`~crappy.tool.image_processing.DICVETool` that
    will perform the image correlation."""

    self.log(logging.INFO, "Instantiating the Disve tool")
    self._disve = DICVETool(patches=self._patches,
                            method=self._method,
                            alpha=self._alpha,
                            delta=self._delta,
                            gamma=self._gamma,
                            finest_scale=self._finest_scale,
                            iterations=self._iterations,
                            gradient_iterations=self._gradient_iterations,
                            patch_size=self._patch_size,
                            patch_stride=self._patch_stride,
                            border=self._border,
                            safe=self._safe,
                            follow=self._follow)

  def loop(self) -> None:
    """This method grabs the latest frame and gives it for processing to the
    :obj:`~crappy.tool.image_processing.DICVETool`. Then sends the result of
    the correlation to the downstream Blocks.
    
    If there's no new frame grabbed, or if the patches were already lost, 
    doesn't do anything. On the first acquired frame, does not process it but 
    initializes the DICVETool with it instead. Also sends the current patches 
    for display to the :class:`~crappy.blocks.camera_processes.Displayer`
    CameraProcess.
    """

    # Do nothing if the patches were already lost
    if not self._lost_patch:
      try:

        # On the first frame, initialize the correlation
        if not self._img0_set:
          self.log(logging.INFO, "Setting the reference image")
          self._disve.set_img0(np.copy(self.img))
          self._img0_set = True
          return

        # Calculating the displacement and sending it to downstream Blocks
        self.log(logging.DEBUG, "Processing the received image")
        data = self._disve.calculate_displacement(self.img)
        self.send([self.metadata['t(s)'], self.metadata, *data])

        # Sending the patches to the Displayer for display
        self.send_to_draw(self._disve.patches)

      # If the patches are lost, deciding whether to raise exception or not
      except RuntimeError as exc:
        if self._raise_on_exit:
          self._logger.exception("Patch exiting the ROI !", exc_info=exc)
          raise
        self._lost_patch = True
        self.log(logging.WARNING, "Patch exiting the ROI, not processing "
                                  "data anymore !")
    
    # If the patches are lost, sleep to avoid spamming the CPU in vain
    else:
      self.fps_count -= 1
      sleep(0.1)

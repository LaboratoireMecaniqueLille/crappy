# coding: utf-8

import numpy as np
from typing import Optional, Union, Literal
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.camera_config import Box, SpotsBoxes
from ...tool.image_processing import DISCorrelTool


class DISCorrelProcess(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can perform
  Dense Inverse Search on a given ROI in the acquired images, and calculate
  various fields from it.

  It is used by the :class:`~crappy.blocks.DISCorrel` Block to parallelize the
  image processing and the image acquisition. It delegates most of the
  computation to the :obj:`~crappy.tool.image_processing.DISCorrelTool`. It is
  from this class that the output values are sent to the downstream Blocks, and
  that the :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` are sent
  to the :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess for
  display.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               patch: Box,
               fields: Optional[list[Union[Literal['x', 'y', 'r', 'exx', 'eyy',
                                                   'exy', 'eyx', 'exy2', 'z'],
                                           np.ndarray]]] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False) -> None:
    """Sets the arguments and initializes the parent class.
    
    Args:
      patch: An instance of the
        :class:`~crappy.tool.camera_config.config_tools.Box` class, containing
        the coordinates of the ROI to perform DIS on. This argument is passed
        to the :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used
        in this class.
      fields: fields: The base of fields to use for the projection, given as a
        :obj:`list` of :obj:`str` or :mod:`numpy` arrays (both types can be
        mixed). Strings are for using automatically-generated fields, the
        available ones are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

        If users provide their own fields as arrays, they will be used as-is to
        run the correlation. The user-provided fields must be of shape:
        ::

          (patch_height, patch_width, 2)

        This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.

        .. versionchanged:: 2.0.5 provided fields can now be numpy arrays
      alpha: Weight of the smoothness term in DISFlow, as a :obj:`float`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      delta: Weight of the color constancy term in DISFlow, as a :obj:`float`.
        This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      gamma: Weight of the gradient constancy term in DISFlow , as a
        :obj:`float`. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DISFlow (`0` means full scale), as an :obj:`int`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      init: If :obj:`True`, the last field is used to initialize the
        calculation for the next one. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DISFlow, as an :obj:`int`. This argument
        is passed to the :obj:`~crappy.tool.image_processing.DISCorrelTool` and
        not used in this class.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DISFlow, as an :obj:`int`. This
        argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      patch_size: Size of an image patch for matching in DISFlow
        (in pixels). This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      patch_stride: Stride between neighbor patches in DISFlow. Must be
        less than patch size. This argument is passed to the
        :obj:`~crappy.tool.image_processing.DISCorrelTool` and not used in this
        class.
      residual: If :obj:`True`, the residuals will be computed at each new
        frame and sent to downstream Blocks, by default under the ``'res'``
        label.
    """

    super().__init__()

    # Arguments to pass to the DISCorrelTool
    self._box = patch
    self._fields = fields
    self._alpha = alpha
    self._delta = delta
    self._gamma = gamma
    self._finest_scale = finest_scale
    self._init = init
    self._iterations = iterations
    self._gradient_iterations = gradient_iterations
    self._patch_size = patch_size
    self._patch_stride = patch_stride
    
    # Other attributes
    self._residual = residual
    self._dis_correl: Optional[DISCorrelTool] = None
    self._img0_set = False

  def init(self) -> None:
    """Instantiates the :obj:`~crappy.tool.image_processing.DISCorrelTool` that
    will perform the Dense Inverse Search."""

    self.log(logging.INFO, "Instantiating the Discorrel tool")
    self._dis_correl = DISCorrelTool(
        box=self._box,
        fields=self._fields,
        alpha=self._alpha,
        delta=self._delta,
        gamma=self._gamma,
        finest_scale=self._finest_scale,
        init=self._init,
        iterations=self._iterations,
        gradient_iterations=self._gradient_iterations,
        patch_size=self._patch_size,
        patch_stride=self._patch_stride)
    self._dis_correl.set_box()

  def loop(self) -> None:
    """This method grabs the latest frame and gives it for processing to the
    :obj:`~crappy.tool.image_processing.DISCorrelTool`. Then sends the result
    of the dense inverse search to the downstream Blocks.

    If there's no new frame grabbed, doesn't do anything. On the first acquired
    frame, does not process it but initializes the DISCorrelTool with it
    instead. Also sends the ROI for display to the
    :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess.
    """

    # On the first frame, initializes the dense inverse search
    if not self._img0_set:
      self.log(logging.INFO, "Setting the reference image")
      self._dis_correl.set_img0(np.copy(self.img))
      self._img0_set = True
      return

    # Calculating the fields and sending them to downstream Blocks
    self.log(logging.DEBUG, "Processing the received image")
    data = self._dis_correl.get_data(self.img, self._residual)
    self.send([self.metadata['t(s)'], self.metadata, *data])

    # Sending the ROI to the Displayer for display
    self.send_to_draw(SpotsBoxes(self._dis_correl.box))

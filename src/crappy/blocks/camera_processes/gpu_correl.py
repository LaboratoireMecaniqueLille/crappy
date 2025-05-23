# coding: utf-8

import numpy as np
from typing import Optional, Union
from pathlib import Path
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.image_processing import GPUCorrelTool


class GPUCorrelProcess(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can perform
  GPU-accelerated digital image correlation on a given mask of the acquired
  images, and calculate various fields from it.

  It is used by the :class:`~crappy.blocks.GPUCorrel` Block to parallelize the
  image processing and the image acquisition. It delegates most of the
  computation to the :class:`~crappy.tool.image_processing.GPUCorrelTool`. It
  is from this class that the output values are sent to the downstream Blocks.

  It is also this class that takes the decision to send or not the results to
  downstream Blocks based on the value of the calculated residuals, if this
  option is enabled by the user.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               discard_limit: float = 3,
               discard_ref: int = 5,
               calc_res: bool = False,
               img_ref: Optional[np.ndarray] = None,
               verbose: int = 0,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               fields: Optional[list[Union[str, np.ndarray]]] = None,
               mask: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """Sets the arguments and initializes the parent class.
    
    Args:
      discard_limit: If ``calc_res`` is :obj:`True`, the result of the
        correlation is not sent to the downstream Blocks if the residuals for
        the current image are greater than ``discard_limit`` times the average
        residual for the last ``discard_ref`` images.
      discard_ref: If ``calc_res`` is :obj:`True`, the result of the
        correlation is not sent to the downstream Blocks if the residuals for
        the current image are greater than ``discard_limit`` times the average
        residual for the last ``discard_ref`` images.
      calc_res: If :obj:`True`, calculates the residuals after performing the
        correlation and returns the residuals along with the correlation data.
        By default, the residuals are returned under the label ``'res'``.
      img_ref: A reference image for the correlation, given as a 2D
        :obj:`numpy.array` with `dtype` :obj:`numpy.float32`. If not given, the
        first acquired frame will be used as the reference image instead. This
        argument is passed to the
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is displayed, and at level `3` so much information
        is displayed that it slows the code down. This argument is passed to
        the :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used
        in this class.
      levels: Number of levels of the pyramid. More levels may help converging
        on images with large strain, but may fail on images that don't contain
        low spatial frequency. Fewer levels mean that the program runs faster.
        This argument is passed to  the
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.
      resampling_factor: The factor by which the resolution is divided between
        each stage of the pyramid. A low factor ensures coherence between the
        stages, but is more computationally intensive. A high factor allows
        reaching a finer detail level, but may lead to a coherence loss between
        the stages. This argument is passed to the
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.
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
      fields: The base of fields to use for the projection, given as a
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
        :class:`~crappy.tool.image_processing.GPUCorrelTool` and not used in
        this class.

        .. versionchanged:: 2.0.5 provided fields can now be numpy arrays
      mask: The mask used for weighting the region of interest on the image. It
        is generally used to prevent unexpected behavior on the border of the
        image. This argument is passed to the
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

    super().__init__()

    # Arguments to pass to the GPUCorrelTool
    self._verbose = verbose
    self._levels = levels
    self._resampling_factor = resampling_factor
    self._kernel_file = kernel_file
    self._iterations = iterations
    self._fields = fields
    self._mask = mask
    self._mul = mul

    # Other attributes
    self._correl: Optional[GPUCorrelTool] = None
    self._img_ref = img_ref
    self._img0_set = img_ref is not None

    self._res_history = [np.inf]
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._calc_res = calc_res

  def init(self) -> None:
    """Initializes the GPUCorrelTool, and set its reference image if a
    ``img_ref`` argument was provided."""

    # Instantiating the GPUCorrelTool
    self.log(logging.INFO, "Instantiating the GPUCorrel tool")
    self._correl = GPUCorrelTool(logger_name=self.name,
                                 context=None,
                                 verbose=self._verbose,
                                 levels=self._levels,
                                 resampling_factor=self._resampling_factor,
                                 kernel_file=self._kernel_file,
                                 iterations=self._iterations,
                                 fields=self._fields,
                                 ref_img=self._img_ref,
                                 mask=self._mask,
                                 mul=self._mul)

    # Setting the reference image if it was given as an argument
    if self._img_ref is not None:
      self.log(logging.INFO, "Initializing the GPUCorrel tool with the "
                             "given reference image")
      self._correl.set_img_size(self._img_ref.shape)
      self._correl.set_orig(self._img_ref.astype(np.float32))
      self.log(logging.INFO, "Preparing the GPUCorrel tool")
      self._correl.prepare()

  def loop(self) -> None:
    """This method grabs the latest frame and gives it for processing to the
    :class:`~crappy.tool.image_processing.GPUCorrelTool`. Then sends the result
    of the correlation to the downstream Blocks.

    If there's no new frame grabbed, doesn't do anything. On the first acquired
    frame, does not process it but initializes the GPUCorrelTool with it
    instead if no reference image was given as argument. If requested by the
    user, also calculates the residuals and checks that they are within the
    provided limit. Otherwise, just drops the calculated data.
    """

    # Setting the reference image with the first received frame if it was not
    # given as an argument
    if not self._img0_set:
      self.log(logging.INFO, "Setting the reference image")
      self._correl.set_img_size(self.img.shape)
      self._correl.set_orig(self.img.astype(np.float32))
      self._correl.prepare()
      self._img0_set = True
      return

    # Performing the image correlation
    self.log(logging.DEBUG, "Processing the received image")
    data = [self.metadata['t(s)'], self.metadata]
    data += self._correl.get_disp(self.img.astype(np.float32)).tolist()

    # Calculating the residuals if requested
    if self._calc_res:
      self.log(logging.DEBUG, "Calculating the residuals")
      res = self._correl.get_res()
      data.append(res)

      # Checking that the residuals are within the given limit, otherwise
      # dropping the calculated data
      if self._discard_limit:
        self.log(logging.DEBUG, "Adding residuals to the residuals "
                                "history")
        self._res_history.append(res)
        self._res_history = self._res_history[-self._discard_ref - 1:]

        if (res > self._discard_limit *
            float(np.average(self._res_history[:-1]))):
          self.log(logging.WARNING, "Residual too high, not sending "
                                    "values")
          return

    # Sending the data to downstream Blocks
    self.send(data)

  def finish(self) -> None:
    """Performs cleanup on the
    :class:`~crappy.tool.image_processing.GPUCorrelTool`."""

    if self._correl is not None:
      self.log(logging.INFO, "Cleaning up the GPUCorrel tool")
      self._correl.clean()

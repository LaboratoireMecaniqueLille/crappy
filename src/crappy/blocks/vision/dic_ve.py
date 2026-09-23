# coding: utf-8

from typing import Literal
from collections.abc import Sequence
import numpy as np
import logging

from .block import VisionBlock, ConfigRequest
from ...tool.image_processing import DICVETool, LostPatchError
from ...tool.camera_config import DICVEConfig, SpotsBoxes


class DICVEProcessor(VisionBlock):
  """Performs video-extensometry by tracking textured image patches.

  Unlike :class:`~crappy.blocks.DICVE`, this Block does not acquire, display,
  or record images itself. It only implements the image-processing stage and
  must receive images from exactly one upstream
  :class:`~crappy.links.img_link.ImageLink`. Image acquisition is normally
  handled by a :class:`~crappy.blocks.vision.CameraSource`. The received images
  must be single-channel, 8-bit :mod:`numpy` arrays.

  Between one and four rectangular patches can be tracked. Their coordinates
  can be supplied directly with ``patches`` or selected interactively in a
  :class:`~crappy.tool.camera_config.DICVEConfig` window opened by the upstream
  image source. When patches are already provided, the configuration request
  is optional and can be disabled with ``request_configuration``. When patches
  are omitted and the source is a
  :class:`~crappy.blocks.vision.CameraSource`, its ``config`` and
  ``allow_downstream_config`` arguments must both be enabled.

  The first received image becomes the fixed reference image and produces no
  output. For every subsequent image, a
  :class:`~crappy.tool.image_processing.DICVETool` measures each patch's
  displacement using the selected correlation method. The tracking windows can
  optionally follow these displacements. With at least two patches, horizontal
  and vertical strain are calculated from the relative displacement of the
  extreme patch centers. With one patch, both strain values remain zero.

  Each processed image produces its timestamp and complete metadata, followed
  by the patch-center coordinates, vertical strain, horizontal strain, and
  patch displacements. Coordinates and displacements are reported as ``(y,
  x)`` pairs. The current patch boxes are additionally published under the
  reserved ``'overlay'`` label. They can be drawn by an
  :class:`~crappy.blocks.vision.ImageDisplayer` receiving both this regular
  :class:`~crappy.links.link.Link` and images from the same source.

  If a patch can no longer be tracked, the Block either raises a
  :class:`~crappy.tool.image_processing.LostPatchError` or remains idle,
  depending on ``raise_on_patch_exit``. In the latter case, the last valid
  measurements, when available, are sent once more with an empty overlay.

  This Block is similar to
  :class:`~crappy.blocks.vision.VideoExtensoProcessor`, which detects and
  tracks contrasted spots instead of textured patches. The
  :class:`~crappy.blocks.vision.DISCorrelProcessor` Block performs dense image
  correlation on a single patch and projects its optical flow onto selected
  fields.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               patches: Sequence[tuple[int, int, int, int]] | None = None,
               labels: str | Sequence[str] | None = None,
               request_configuration: bool = True,
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
               raise_on_patch_exit: bool = True,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 200) -> None:
    """Sets the correlation and Block options.

    Args:
      patches: Coordinates of the patches to track, each given as ``(y, x,
        height, width)``. Position values must be non-negative integers and
        dimensions must be strictly positive. Between one and four patches can
        be supplied. If omitted, patches must be obtained through upstream
        interactive configuration.
      labels: The six labels carrying the image timestamp, complete metadata,
        patch-center coordinates, vertical strain, horizontal strain, and
        patch displacements, in that order. If omitted, they default to
        ``'t(s)'``, ``'meta'``, ``'Coord(px)'``, ``'Eyy(%)'``, ``'Exx(%)'``,
        and ``'Disp(px)'``. Custom labels must all be supplied at once. The
        reserved ``'overlay'`` label is appended automatically and must not be
        included.
      request_configuration: If :obj:`True`, asks the upstream image source to
        display a :class:`~crappy.tool.camera_config.DICVEConfig` window. The
        request is required when ``patches`` is omitted and optional when
        patches were supplied. If :obj:`False`, no request is made and at least
        one patch must be provided.
      method: Correlation method used to measure patch displacement. The
        available methods are ``'Disflow'``, ``'Lucas Kanade'``,
        ``'Pixel precision'``, and ``'Parabola'``. DISFlow and Lucas-Kanade use
        the corresponding OpenCV optical-flow implementations. Pixel precision
        locates the peak of a Fourier-domain cross-correlation with one-pixel
        resolution, while Parabola refines that peak to sub-pixel resolution.
      alpha: Weight of the DISFlow smoothness term. It must be finite and
        non-negative. Ignored by the other correlation methods.
      delta: Weight of the DISFlow color-constancy term. It must be finite and
        non-negative. Ignored by the other correlation methods.
      gamma: Weight of the DISFlow gradient-constancy term. It must be finite
        and non-negative. Ignored by the other correlation methods.
      finest_scale: Finest Gaussian-pyramid level on which DISFlow computes
        optical flow. Zero selects the original image resolution. Ignored by
        the other correlation methods.
      iterations: Number of fixed-point iterations of DISFlow variational
        refinement per scale. Set to zero to disable variational refinement.
        Ignored by the other correlation methods.
      gradient_iterations: Maximum number of gradient-descent iterations in
        the DISFlow patch inverse-search stage. Ignored by the other
        correlation methods.
      patch_size: Size of the image patches matched internally by DISFlow, in
        pixels. It must be a strictly positive integer. Ignored by the other
        correlation methods.
      patch_stride: Stride between neighboring DISFlow patches, in pixels. It
        must be strictly positive and smaller than ``patch_size``. Lower values
        generally improve flow quality at the cost of computation time.
        Ignored by the other correlation methods.
      border: Fraction of each tracked patch excluded across each axis before
        averaging its DISFlow displacement. For example, ``0.2`` retains the
        central 80 percent of the flow field. It must be at least zero and
        strictly less than one. Ignored by the other correlation methods.
      safe: If :obj:`True`, checks that the tracked patches remain inside the
        image and reports a lost patch otherwise. Disabling this check can lead
        to unexpected behavior when a patch leaves the image.
      follow: If :obj:`True`, moves each tracking window according to its
        previous displacement. This is recommended when the displacement
        between frames can be large relative to the patch size.
      raise_on_patch_exit: If :obj:`True`, propagates
        :class:`~crappy.tool.image_processing.LostPatchError` when a patch can
        no longer be tracked. If :obj:`False`, processing stops while the rest
        of the test continues; the last valid measurements are sent with an
        empty overlay when available.
      display_freq: If :obj:`True`, periodically displays the image-processing
        frequency.
      debug: If :obj:`True`, displays all log messages, including
        :obj:`~logging.DEBUG` messages. If :obj:`False`, only displays messages
        at :obj:`~logging.INFO` level or higher. If :obj:`None`, disables
        logging for this Block.
      freq: Target looping frequency for this Block. If :obj:`None`, loops as
        fast as possible. This is an upper bound for checking and processing
        newly received images, not the acquisition frequency of the source.
    """

    super().__init__(img_shape=None,
                     img_dtype=None,
                     display_freq=display_freq,
                     debug=debug,
                     freq=freq)

    # Make sure the patches are correctly provided
    if patches is not None and len(patches) > 4:
      raise ValueError("Only 1 to 4 patches can be provided")
    if (patches is not None and patches
        and (not all(isinstance(patch, tuple) for patch in patches)
             or any(len(patch) != 4 for patch in patches)
             or any(not all(isinstance(val, int) for val in patch)
                    for patch in patches)
             or any(not all(val >= 0 for val in patch) for patch in patches))):
      raise ValueError("The patches should be provided as a sequence of "
                       "tuples of 4 positive integer values")
    if patches is not None and patches and any((patch[2] <= 0 or patch[3] <= 0)
                                               for patch in patches):
      raise ValueError("The width and height of the patches must be "
                       "strictly positive integers")

    # Forcing the labels into a list
    if labels is None:
      _labels: list[str] = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)',
                            'Exx(%)', 'Disp(px)']
    elif isinstance(labels, str):
      _labels: list[str] = [labels]
    else:
      _labels: list[str] = list(labels)

    # Making sure a consistent number of labels and fields was given
    if len(_labels) != 6:
      raise ValueError("The number of labels should be 6 !\n"
                       "Make sure that the time label was given")

    # Adding the reserved overlay label
    _labels.append('overlay')
    self.labels = _labels

    # Checking the validity of the provided arguments
    if not isinstance(request_configuration, bool):
      raise TypeError("request_configuration must be a boolean")
    if not request_configuration and (patches is None or not patches):
      raise ValueError("A patch must be provided if request_configuration is "
                       "set to False")
    if method not in ('Disflow', 'Lucas Kanade',
                      'Pixel precision', 'Parabola'):
      raise ValueError("The method argument should be one of 'Disflow', "
                       "'Lucas Kanade', 'Pixel precision', 'Parabola'")
    if (not isinstance(alpha, (float, int, bool)) or not np.isfinite(alpha)
        or alpha < 0):
      raise ValueError("alpha must be a finite, non-negative number")
    if (not isinstance(delta, (float, int, bool)) or not np.isfinite(delta)
        or delta < 0):
      raise ValueError("delta must be a finite, non-negative number")
    if (not isinstance(gamma, (float, int, bool)) or not np.isfinite(gamma)
        or gamma < 0):
      raise ValueError("gamma must be a finite, non-negative number")
    if not isinstance(finest_scale, int) or finest_scale < 0:
      raise ValueError("finest_scale must be a non-negative integer")
    if not isinstance(iterations, int) or iterations < 0:
      raise ValueError("iterations must be a non-negative integer")
    if not isinstance(gradient_iterations, int) or gradient_iterations < 0:
      raise ValueError("gradient_iterations must be a non-negative integer")
    if not isinstance(patch_size, int) or patch_size <= 0:
      raise ValueError("patch_size must be a positive integer")
    if not isinstance(patch_stride, int) or patch_stride <= 0:
      raise ValueError("patch_stride must be a positive integer")
    if patch_stride >= patch_size:
      raise ValueError("patch_stride must be strictly less than patch_size")
    if ((not isinstance(border, float) and not isinstance(border, int))
        or not 0 <= border < 1):
      raise ValueError("border must be greater than or equal to 0 and strictly"
                       " less than 1")
    if not isinstance(safe, bool):
      raise TypeError("safe must be a boolean")
    if not isinstance(follow, bool):
      raise TypeError("follow must be a boolean")
    if not isinstance(raise_on_patch_exit, bool):
      raise TypeError("raise_on_patch_exit must be a boolean")

    # These arguments are for the DICVETool
    self._method: Literal['Disflow', 'Lucas Kanade',
                          'Pixel precision', 'Parabola'] = method
    self._alpha: float = alpha
    self._delta: float = delta
    self._gamma: float = gamma
    self._finest_scale: int = finest_scale
    self._iterations: int = iterations
    self._gradient_iterations: int = gradient_iterations
    self._patch_size: int = patch_size
    self._patch_stride: int = patch_stride
    self._border: float = border
    self._safe: bool = safe
    self._follow: bool = follow

    # Other attributes
    self._request_configuration: bool = request_configuration
    self._raise_on_exit: bool = raise_on_patch_exit
    self._disve: DICVETool | None = None
    self._img0_set: bool = False
    self._lost_patch: bool = False
    self._last_data: tuple[list[tuple[float | int, float | int]],
                           float | int, float | int,
                           list[tuple[float | int, float | int]]] | None = None

    # Set the patches if already provided by user
    self._patches: SpotsBoxes = SpotsBoxes()
    if patches is not None and patches:
      self._patches.set_spots(list(patches))
      self._patches.save_length()

  def prepare(self) -> None:
    """Receives the source configuration and initializes patch correlation.

    This method checks that the Block has exactly one input ImageLink, no input
    regular Link, and no output ImageLink. It then receives any requested
    upstream configuration, creates the DICVE tool with the selected patches,
    and retrieves the shared image buffer from the source.

    Raises:
      IOError: If the Block's Link topology is unsupported.
      RuntimeError: If neither configured nor user-provided patches are
        available, or if required startup objects are unavailable.
      NotImplementedError: If more than one image source returns configuration
        data.
      TypeError: If received configuration data cannot be unpacked as expected.
      ValueError: If received configuration data has an invalid number of
        values.
    """

    # Ensuring Link consistency
    if not self.img_inputs:
      raise IOError("This VisionBlock is useless without an input ImageLink")
    if len(self.img_inputs) > 1:
      raise IOError("This VisionBlock accepts exactly one input ImageLink")
    if self.img_outputs:
      raise IOError("This VisionBlock does not support output ImageLink")
    if self.inputs:
      raise IOError("This Block does not accept input Links")

    # Receive the configuration information from upstream Blocks
    configs = self.recv_configs()
    valid = [config for config in configs.values() if config is not None]
    if not len(valid) and self._patches.empty():
      raise RuntimeError("The patches to track weren't provided and no "
                         "configuration information was received from upstream"
                         "Blocks, cannot proceed!\nEither something went "
                         "wrong, or you did not provide the patches "
                         "coordinates")
    elif len(valid) > 1:
      raise NotImplementedError("Ambiguous situation with at least two "
                                "configurations received from upstream "
                                "Blocks, don't know how to handle")
    for source, config in configs.items():
      if config is not None:
        try:
          self._patches, = config
        except (ValueError, TypeError):
          self.log(logging.ERROR, f"Got invalid configuration data from "
                                  f"Block {source}")
          raise

    # Catch uninitialized attributes early
    if self._log_queue is None:
      raise RuntimeError("At that point the log_queue should be set but it "
                         "isn't")
    if self._patches is None or self._patches.empty():
      raise RuntimeError("At that point the patches to track should be set but"
                         "they are not")

    self.log(logging.INFO, "Instantiating the DICVE tool")
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

    # Mandatory otherwise the Block won't run
    super().prepare()

  def loop(self) -> None:
    """Processes the newest image and sends strain data and patch overlays.

    If no new image is available, this method returns immediately. The first
    received image is stored as the correlation reference and produces no
    output. Each later image is correlated against that reference, then its
    timestamp, metadata, patch coordinates, strains, displacements, and current
    patch overlay are sent through the regular output Links.

    When patch loss is tolerated, the final valid measurements are sent once
    more with an empty overlay iterable, then the Block remains idle. If no
    valid measurement was produced before the loss, no final data point is
    sent.

    Raises:
      LostPatchError: If a patch can no longer be tracked and
        ``raise_on_patch_exit`` is :obj:`True`.
      RuntimeError: If required processing objects are unavailable.
    """

    # If the patch exited but raise_on_exit is False, do nothing
    if self._lost_patch:
      self.log(logging.DEBUG, "Patches were lost, remaining idle")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return

    # Nothing to do if no new image was received
    if not (upd_links := self.receive_imgs()):
      self.log(logging.DEBUG, "No new image received during this loop")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Get the ImageLink name
    upd_link, = upd_links

    # Handles to the received data
    metadata = self.last_received[upd_link].metadata
    if metadata is None:
      raise RuntimeError("At that point, the image metadata should not be "
                         "empty")
    img = self.last_received[upd_link].img

    try:
      # On the first frame, initialize the correlation
      if not self._img0_set:
        if self._disve is None:
          raise RuntimeError("The DICVETool should have been instantiated")
        self.log(logging.INFO, "Setting the reference image")
        self._disve.set_img0(np.copy(img))
        self._img0_set = True
        # If requested, displays the FPS of the image display
        if self.display_freq:
          self._print_freq(img_handled=True)
        return

      # Processing the received frame
      self.log(logging.DEBUG, "Processing the received image")

      if self._disve is None:
        raise RuntimeError("The DICVE tool should have been instantiated")

      # Sending the results to the downstream Blocks, including the overlay
      data = self._disve.calculate_displacement(img)
      overlay = self._disve.patches.copy(
          use_displacements=not self._follow)
      self.send([metadata['t(s)'], metadata, *data, overlay])

      # Save the last data points for the case when patches are lost
      self._last_data = data

    # If the patches are lost, raise or not depending on arguments
    except LostPatchError as exc:
      self.log(logging.WARNING, f"No longer processing data because a patch "
                                f"was lost: {exc}")
      self._lost_patch = True
      if self._raise_on_exit:
        raise

      # Send a message with the last data points and no overlay for stopping
      # the display of the overlay
      if self._last_data is not None:
        self.send([metadata['t(s)'], metadata, *self._last_data, list()])

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def request_config(self, source: str) -> ConfigRequest | None:
    """Builds the DICVE configuration request for an image source.

    Args:
      source: Name of the upstream image source that should run the
        configuration window.

    Returns:
      A request for :class:`~crappy.tool.camera_config.DICVEConfig`, or
      :obj:`None` when upstream configuration is disabled. The request contains
      the current patch boxes and is flagged as required only when no patches
      were provided.
    """

    if not self._request_configuration:
      return None

    return ConfigRequest(requester=self.name,
                         args=tuple(),
                         kwargs={'patches': self._patches},
                         configurator=DICVEConfig,
                         img_source=source,
                         required=self._patches.empty())

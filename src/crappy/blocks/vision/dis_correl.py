# coding: utf-8

from typing import Literal, Sequence
import numpy as np
import logging

from .block import VisionBlock, ConfigRequest
from ...tool.image_processing.fields import allowed_fields
from ...tool.image_processing import DISCorrelTool
from ...tool.camera_config import DISCorrelConfig, Box, SpotsBoxes

field_type = Literal['x', 'y', 'r', 'exx', 'eyy',
                     'exy', 'eyx', 'exy2', 'z'] | np.ndarray


class DISCorrelProcessor(VisionBlock):
  """Performs Dense Inverse Search correlation on a stream of images.

  Unlike :class:`~crappy.blocks.DISCorrel`, this Block does not acquire,
  display, or record images itself. It only implements the image-processing
  stage and must receive images from exactly one upstream
  :class:`~crappy.links.img_link.ImageLink`. Image acquisition is normally
  handled by a :class:`~crappy.blocks.vision.CameraSource`. The received images
  must be single-channel, 8-bit :mod:`numpy` arrays.

  The correlation is performed on one rectangular patch. Its coordinates can
  be supplied directly with ``patch`` or selected interactively in a
  :class:`~crappy.tool.camera_config.DISCorrelConfig` window opened by the
  upstream image source. When a patch is already provided, the configuration
  request is optional. Set ``request_configuration`` to :obj:`False` to
  suppress this request entirely.

  The first received image becomes the fixed reference image and produces no
  output. For every subsequent image, a
  :class:`~crappy.tool.image_processing.DISCorrelTool` calculates the dense
  optical flow relative to that reference and projects it onto the requested
  fields. The available generated fields cover translations, rotation, normal
  and shear strains, and zoom. User-defined vector fields can also be supplied
  as :mod:`numpy` arrays.

  Each processed image produces its timestamp and complete metadata followed
  by one scalar per requested field and, optionally, the correlation residual.
  The selected patch is additionally published under the reserved
  ``'overlay'`` label. It can be drawn by an
  :class:`~crappy.blocks.vision.ImageDisplayer` receiving both this regular
  :class:`~crappy.links.link.Link` and images from the same source.

  This Block is similar to :class:`~crappy.blocks.GPUCorrel`, which performs
  image correlation using GPU acceleration. The
  :class:`~crappy.blocks.DICVE` Block instead tracks multiple textured patches
  and derives strain from their relative displacement.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               patch: tuple[int, int, int, int] | None = None,
               fields: field_type | Sequence[field_type] | None = None,
               labels: str | Sequence[str] | None = None,
               request_configuration: bool = True,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,
               border: int | tuple[int, int] | None = 16,
               follow: bool = False,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 200) -> None:
    """Sets the correlation and Block options.

    Args:
      patch: Coordinates of the patch on which to perform correlation, given
        as ``(y, x, height, width)``. The position values must be non-negative
        integers and the dimensions must be strictly positive. If omitted, a
        patch must be obtained through upstream interactive configuration.
      fields: Fields onto which the calculated optical flow is projected. A
        field can be one of ``'x'``, ``'y'``, ``'r'``, ``'exx'``, ``'eyy'``,
        ``'exy'``, ``'eyx'``, ``'exy2'``, or ``'z'``. It can also be a custom
        numeric :mod:`numpy` array of shape
        ``(patch_height, patch_width, 2)`` containing finite values and having
        a nonzero norm. Strings and arrays can be mixed in an iterable. If
        omitted, the fields default to ``'x'``, ``'y'``, ``'exx'``, and
        ``'eyy'``.
      labels: Labels carrying the image timestamp, complete metadata, and one
        scalar value per requested field, in that order. With the default
        fields, these default to ``'t(s)'``, ``'meta'``, ``'x(pix)'``,
        ``'y(pix)'``, ``'Exx(%)'``, and ``'Eyy(%)'``. When custom fields are
        provided, all corresponding labels must also be supplied. The
        automatically added ``'res'`` and reserved ``'overlay'`` labels must
        not be included.
      request_configuration: If :obj:`True`, asks the upstream image source to
        display a :class:`~crappy.tool.camera_config.DISCorrelConfig` window.
        The request is required when ``patch`` is omitted and optional when a
        patch was supplied. If :obj:`False`, no request is made and ``patch``
        must be provided.
      alpha: Weight of the smoothness term in DISFlow. Must be finite and
        non-negative.
      delta: Weight of the color-constancy term in DISFlow. Must be finite and
        non-negative.
      gamma: Weight of the gradient-constancy term in DISFlow. Must be finite
        and non-negative.
      finest_scale: Finest Gaussian-pyramid level on which DISFlow computes
        the optical flow. Zero selects the original image resolution.
      iterations: Number of fixed-point iterations of variational refinement
        per scale. Set to zero to disable variational refinement.
      gradient_iterations: Maximum number of gradient-descent iterations in
        the patch inverse-search stage.
      init: If :obj:`True`, uses the previously calculated optical flow to
        initialize the next calculation.
      patch_size: Size of the image patches matched by DISFlow, in pixels. It
        must be a strictly positive integer.
      patch_stride: Stride between neighboring DISFlow patches, in pixels. It
        must be strictly positive and smaller than ``patch_size``. Lower
        values generally improve flow quality at the cost of computation time.
      residual: If :obj:`True`, calculates the average absolute optical-flow
        residual and sends it under the automatically added ``'res'`` label.
      border: Width in pixels of the additional area around the correlation
        patch that is passed to DISFlow. An :obj:`int` applies the same border
        in both directions, while a ``(x, y)`` tuple allows setting the
        horizontal and vertical borders independently. ``None`` uses the full
        image, which corresponds to the legacy behavior. A larger border
        provides stability under large displacements, at the cost of a
        performance penalty. Without ``follow``, it should exceed the maximum
        displacement from the reference image; with ``follow``, it should
        exceed the expected displacement between consecutive frames.

        ..  versionadded:: 2.1.0
      follow: If :obj:`True`, shifts the correlation area according to the
        average rigid-body displacement measured on the patch. This allows the
        patch to follow large cumulative translations while keeping the
        correlation area small. The reported fields remain relative to the
        original reference image. This option is relevant when large
        displacements of the observed area are expected.

        .. versionadded:: 2.1.0
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
    if patch is not None and (not isinstance(patch, tuple)
                              or len(patch) != 4
                              or not all(isinstance(val, int) for val in patch)
                              or not all(val >= 0 for val in patch)):
      raise ValueError("The patch should be provided as a tuple of 4 "
                       "positive integer values")
    if patch is not None and (patch[2] <= 0 or patch[3] <= 0):
      raise ValueError("The width and height of the patch must be "
                       "strictly positive integers")

    # Make sure that proper labels are provided if custom fields are provided
    if fields is not None and labels is None:
      raise ValueError("Custom fields were provided but no custom labels!")

    # Forcing the fields into a list
    if fields is None:
      _fields = ['x', 'y', 'exx', 'eyy']
    elif isinstance(fields, str) or isinstance(fields, np.ndarray):
      _fields = [fields]
    else:
      _fields = list(fields)

    if not all(isinstance(field, (np.ndarray, str)) for field in _fields):
      raise TypeError("All the provided fields must be either strings or "
                      "numpy arrays")
    if not _fields:
      raise ValueError("At least one field must be provided")
    if not all(field in allowed_fields for field in _fields
               if isinstance(field, str)):
      raise ValueError(f"The only allowed values for the fields given as "
                       f"strings are {allowed_fields}")

    # Forcing the labels into a list
    if labels is None:
      _labels: list[str] = ['t(s)', 'meta', 'x(pix)', 'y(pix)',
                            'Exx(%)', 'Eyy(%)']
    elif isinstance(labels, str):
      _labels: list[str] = [labels]
    else:
      _labels: list[str] = list(labels)

    if not isinstance(residual, bool):
      raise TypeError("residual must be a boolean")

    # Adding the residuals if required
    if residual and _labels is not None:
      _labels.append('res')

    # Making sure a consistent number of labels and fields was given
    if 2 + len(_fields) + int(residual) != len(_labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time label was given")

    # Adding the reserved overlay label
    _labels.append('overlay')
    self.labels = _labels

    if not isinstance(request_configuration, bool):
      raise TypeError("request_configuration must be a boolean")
    if not request_configuration and patch is None:
      raise ValueError("A patch must be provided if request_configuration is "
                       "set to False")
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
    if not isinstance(init, bool):
      raise TypeError("init must be a boolean")
    if not isinstance(patch_size, int) or patch_size <= 0:
      raise ValueError("patch_size must be a positive integer")
    if not isinstance(patch_stride, int) or patch_stride <= 0:
      raise ValueError("patch_stride must be a positive integer")
    if patch_stride >= patch_size:
      raise ValueError("patch_stride must be strictly less than patch_size")
    if border is not None and not isinstance(border, (int, tuple)):
      raise TypeError("border must be either None, an integer, or a tuple of "
                      "two integers")
    if (isinstance(border, tuple) and
        (len(border) != 2 or not all(isinstance(val, int) for val in border) or
         not all(val >= 0 for val in border))):
      raise ValueError("If provided as a tuple, border must contain exactly "
                       "two non-negative integers")
    if isinstance(border, int) and border < 0:
      raise ValueError("If provided as an integer, border must be "
                       "non-negative")
    if not isinstance(follow, bool):
      raise TypeError("follow must be a boolean")

    # These arguments are for the DISCorrelTool
    self._fields: list[Literal['x', 'y', 'r', 'exx', 'eyy',
                               'exy', 'eyx', 'exy2', 'z'] |
                       np.ndarray] = _fields
    self._alpha: float = alpha
    self._delta: float = delta
    self._gamma: float = gamma
    self._finest_scale: int = finest_scale
    self._init: bool = init
    self._iterations: int = iterations
    self._gradient_iterations: int = gradient_iterations
    self._patch_size: int = patch_size
    self._patch_stride: int = patch_stride
    self._border: int | tuple[int, int] | None = border
    self._follow: bool = follow

    # Other attributes
    self._request_configuration: bool = request_configuration
    self._residual: bool = residual
    self._dis_correl: DISCorrelTool | None = None
    self._img0_set: bool = False

    # Instantiating the Box containing the patch to track
    if patch is not None:
      self._patch: Box = Box(x_start=patch[1],
                             x_end=patch[1] + patch[3],
                             y_start=patch[0],
                             y_end=patch[0] + patch[2])
    else:
      self._patch: Box = Box()

  def prepare(self) -> None:
    """Receives the source configuration and initializes correlation.

    This method checks that the Block has exactly one input ImageLink, no input
    regular Link, and no output ImageLink. It then receives any requested
    upstream configuration, creates the DIS correlation tool, retrieves the
    shared image buffer, and prepares the projection fields for the selected
    patch.

    Raises:
      IOError: If the Block's Link topology is unsupported.
      RuntimeError: If neither a configured nor a user-provided patch is
        available, or if required startup objects are unavailable.
      NotImplementedError: If more than one image source returns configuration
        data.
      ValueError: If the selected patch or a custom projection field is
        invalid.
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
    if not len(valid) and self._patch.no_points():
      raise RuntimeError("The patch to track wasn't provided and no "
                         "configuration information was received from upstream"
                         "Blocks, cannot proceed!\nEither something went "
                         "wrong, or you did not provide the patch coordinates")
    elif len(valid) > 1:
      raise NotImplementedError("Ambiguous situation with at least two "
                                "configurations received from upstream "
                                "Blocks, don't know how to handle")
    for source, config in configs.items():
      if config is not None:
        try:
          self._patch, = config
        except (ValueError, TypeError):
          self.log(logging.ERROR, f"Got invalid configuration data from "
                                  f"Block {source}")
          raise

    # Catch uninitialized attributes early
    if self._log_queue is None:
      raise RuntimeError("At that point the log_queue should be set but it "
                         "isn't")
    if self._patch is None or self._patch.no_points():
      raise RuntimeError("At that point the patch to track should be set but "
                         "it is not")

    self.log(logging.INFO, "Instantiating the DISCorrel tool")
    self._dis_correl = DISCorrelTool(
        box=self._patch,
        fields=self._fields,
        alpha=self._alpha,
        delta=self._delta,
        gamma=self._gamma,
        finest_scale=self._finest_scale,
        init=self._init,
        iterations=self._iterations,
        gradient_iterations=self._gradient_iterations,
        patch_size=self._patch_size,
        patch_stride=self._patch_stride,
        border=self._border,
        follow=self._follow)

    # Mandatory otherwise the Block won't run
    super().prepare()

    if self._dis_correl is None:
      raise RuntimeError("The DISCorrelTool wasn't properly set")
    self._dis_correl.set_box()

  def loop(self) -> None:
    """Processes the newest image and sends correlation data and an overlay.

    If no new image is available, this method returns immediately. The first
    received image is stored as the correlation reference and is not sent as a
    result. Each later image is correlated against that fixed reference, then
    its timestamp, metadata, projected field values, optional residual, and
    current patch overlay are sent through the regular output Links.
    """

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

    # On the first frame, initializes the dense inverse search
    if not self._img0_set:
      if self._dis_correl is None:
        raise RuntimeError("The DISCorrel tool should have been instantiated")
      self.log(logging.INFO, "Setting the reference image")
      self._dis_correl.set_img0(np.copy(img))
      self._img0_set = True
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=True)
      return

    # Calculating the fields and sending them to downstream Blocks
    if self._dis_correl is None:
      raise RuntimeError("The DISCorrel tool should have been instantiated")
    self.log(logging.DEBUG, "Processing the received image")
    data = self._dis_correl.get_data(img, self._residual)
    x_offset, y_offset = self._dis_correl.offset
    self.send([metadata['t(s)'], metadata, *data,
               SpotsBoxes(self._dis_correl.box + (x_offset, y_offset))])

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def request_config(self, source: str) -> ConfigRequest | None:
    """Builds the DIS correlation configuration request for an image source.

    Args:
      source: Name of the upstream image source that should run the
        configuration window.

    Returns:
      A request for :class:`~crappy.tool.camera_config.DISCorrelConfig`, or
      :obj:`None` when upstream configuration is disabled. The request is
      flagged as required only when no complete patch was provided.
    """

    if not self._request_configuration:
      return None

    return ConfigRequest(requester=self.name,
                         args=tuple(),
                         kwargs={'patch': self._patch},
                         configurator=DISCorrelConfig,
                         img_source=source,
                         required=self._patch.no_points())

# coding: utf-8

from typing import Optional, Union, Literal
from collections.abc import Iterable, Callable
import numpy as np
from pathlib import Path

from .camera_processes import DISCorrelProcess
from .camera import Camera
from ..tool.camera_config import DISCorrelConfig, Box
from .._global import CameraConfigError

field_type = Union[Literal['x', 'y', 'r', 'exx', 'eyy',
                           'exy', 'eyx', 'exy2', 'z'], np.ndarray]


class DISCorrel(Camera):
  """This Block can perform Dense Inverse Search on a sub-frame (patch) of
  images acquired by a :class:`~crappy.camera.Camera` object, and project the
  result on various fields.

  It is mostly used for computing the displacement and the strain over the
  given patch, but other fields are also available. refer to the ``fields`` and
  ``labels`` arguments for more details. It relies on OpenCV's DISFlow
  algorithm, and offers the possibility to adjust many of its settings.

  This Block takes no input :class:`~crappy.links.Link` in a majority of
  situations, and outputs the results of image correlation. It is a subclass of
  the :class:`~crappy.blocks.Camera` Block, and inherits of all its features.
  That includes the possibility to record and to display images in real-time,
  simultaneously to the image acquisition and processing. Refer to the
  documentation of the Camera Block for more information on these features.

  This Block is very similar to the :class:`GPUCorrel` Block, except this
  latter uses GPU-acceleration to perform the image correlation and does not
  use DISFlow. The :class:`~crappy.blocks.DICVE` Block also relies on image
  correlation for computing the displacement and strain on images, but it
  tracks multiple patches and uses video-extensometry.

  Similar to the :class:`~crappy.tool.camera_config.CameraConfig` window that
  can be displayed by the Camera Block, this Block can display a
  :class:`~crappy.tool.camera_config.DISCorrelConfig` window before the test
  starts. Here, the user can also select the patch to track if it was not
  already specified as an argument.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[Literal['cv2', 'mpl']] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               display_freq: bool = False,
               freq: Optional[float] = 200,
               debug: Optional[bool] = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[Literal['sitk', 'pil', 
                                              'cv2', 'npy']] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
               patch: Optional[tuple[int, int, int, int]] = None,
               fields: Union[field_type, Iterable[field_type]] = None,
               labels: Union[str, Iterable[str]] = None,
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
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      camera: The name of the :class:`~crappy.camera.Camera` object to use for
        acquiring the images. Arguments can be passed to this Camera as
        ``kwargs`` of this Block. This argument is ignored if the
        ``image_generator`` argument is provided.
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output. Allows applying a post-processing
        operation to the acquired images. This is done right after the
        acquisition, so the original image is permanently lost and only the
        transformed image is displayed and/or saved and/or further processed.
        The transform operation is not parallelized, so it might negatively
        affect the acquisition framerate if it is too heavy.

        .. versionadded:: 1.5.10
      config: If :obj:`True`, a
        :class:`~crappy.tool.camera_config.DISCorrelConfig` window is displayed
        before the test starts. There, the user can interactively adjust the
        different
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
        available for the selected :class:`~crappy.camera.Camera`, visualize
        the acquired images, and select the patch to track if it hasn't
        been given in the ``patch`` argument. The test starts when closing
        the configuration window. If not enabled, the ``img_dtype``,
        ``img_shape`` and ``patch`` arguments must be provided.

        .. versionadded:: 1.5.10
      display_images: If :obj:`True`, displays the acquired images in a
        dedicated window, using the backend given in ``displayer_backend`` and
        at the frequency specified in ``displayer_framerate``. This option
        should be considered as a debug or basic follow-up feature, it is not
        intended to be very fast nor to display high-quality images. The
        maximum resolution of the displayed images in `640x480`, the images
        might be downscaled to fit in this format. In addition to the acquired
        frames, the tracked patch is also displayed on the image as an
        overlay.

        .. versionchanged:: 1.5.10
           renamed from *show_image* to *display_images*
      displayer_backend: The backend to use for displaying the images. Can be
        either ``'cv2'`` or ``'mpl'``, to use respectively :mod:`cv2` (OpenCV)
        or :mod:`matplotlib`. ``'cv2'`` usually allows achieving a higher
        display frequency. Ignored if ``display_images`` is :obj:`False`. If
        not given and ``display_images`` is :obj:`True`, ``'cv2'`` is tried
        first and ``'mpl'`` second, and the first available one is used.

        .. versionadded:: 1.5.10
      displayer_framerate: The maximum update frequency of the image displayer,
        as an :obj:`int`. This value usually lies between 5 and 30Hz, the
        default is 5. The achieved update frequency might be lower than
        requested. Ignored if ``display_images`` is :obj:`False`.

        .. versionadded:: 1.5.10
      software_trig_label: The name of a label used as a software trigger for
        the :class:`~crappy.camera.Camera`. If given, images will only be
        acquired when receiving data over this label. The received value does
        not matter. This software trigger is not meant to be very precise, it
        is recommended not to rely on it for a trigger frequency greater than
        10Hz, in which case a hardware trigger should be preferred if available
        on the camera.

        .. versionadded:: 2.0.0
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.

        .. versionadded:: 1.5.10
      save_images: If :obj:`True`, the acquired images are saved to the folder
        specified in ``save_folder``, in the format specified in
        ``img_extension``, using the backend specified in ``save_backend``, and
        at the frequency specified in ``save_period``. Each image is saved with
        the name : ``<frame_nr>_<timestamp>.<extension>``, and can thus easily
        be identified. Along with the images, a ``metadata.csv`` file records
        the metadata of all the saved images. This metadata is either the one
        returned by the :meth:`~crappy.camera.Camera.get_image` method of the
        :class:`~crappy.camera.Camera` object, or the default one generated in
        the :meth:`~crappy.blocks.Camera.loop` method of the
        :class:`~crappy.blocks.Camera` Block. Depending on the framerate of the
        camera and the performance of the computer, it is not guaranteed that
        all the acquired images will be recorded.

        .. versionadded:: 1.5.10
      img_extension: The file extension for the recorded images, as a
        :obj:`str` and without the dot. Common file extensions include `tiff`,
        `png`, `jpg`, etc. Depending on the used ``save_backend``, some
        extensions might not be available. It is currently not possible to
        customize the save parameters further than choosing the file extension.
        Ignored if ``save_images`` is :obj:`False`.

        .. versionadded:: 2.0.0
      save_folder: Path to the folder where to save the images, either as a
        :obj:`str` or as a :obj:`pathlib.Path`. Can be an absolute or a
        relative path, pointing to a folder. If the folder does not exist, it
        will be created (if the user has permission). If the given folder
        already contains a ``metadata.csv`` file (and thus likely images from
        Crappy), images are saved to another folder with the same name except
        a suffix is appended. Ignored if ``save_images`` is :obj:`False`. If
        not provided and ``save_images`` is :obj:`True`, the images are saved
        to the folder ``Crappy_images``, created next to the running script.

        .. versionadded:: 1.5.10
      save_period: Must be given as an :obj:`int`. Only one out of that number
        images at most will be saved. Allows to have a known periodicity in
        case the framerate is too high to record all the images. Or simply to
        reduce the number of recorded images if saving them all is not needed.
        Ignored if ``save_images`` is :obj:`False`.

        .. versionadded:: 1.5.10
      save_backend: If ``save_images`` is :obj:`True`, the backend to use for
        recording the images. It should be one of:
        ::

          'sitk', 'pil', 'cv2', 'npy'

        They correspond to the modules :mod:`SimpleITK`, :mod:`PIL` (Pillow
        Fork), :mod:`cv2` (OpenCV), and :mod:`numpy`. Note that the ``'npy'``
        backend saves the images as raw :obj:`numpy.array`, and thus ignores
        the ``img_extension`` argument. Depending on the machine, some backends
        may be faster or slower. For using each backend, the corresponding
        Python module must of course be installed. If not provided and
        ``save_images`` is :obj:`True`, the backends are tried in the same
        order as given above and the first available one is used. ``'npy'`` is
        always available.

        .. versionadded:: 1.5.10
      image_generator: A callable taking two :obj:`float` as arguments and
        returning an image as a :obj:`numpy.array`. **This argument is intended
        for use in the examples of Crappy, to apply an artificial strain on a
        base image. Most users should ignore it.** When given, the ``camera``
        argument is ignored and the images are acquired from the generator. To
        apply a strain on the image, strain values (in `%`) should be sent to
        the Camera Block over the labels ``'Exx(%)'`` and ``'Eyy(%)'``.

        .. versionadded:: 1.5.10
      img_shape: The shape of the images returned by the
        :class:`~crappy.camera.Camera` object as a :obj:`tuple` of :obj:`int`.
        It should correspond to the value returned by :obj:`numpy.shape`.
        **This argument is mandatory in case** ``config`` **is** :obj:`False`.
        It is otherwise ignored.

        .. versionadded:: 2.0.0
      img_dtype: The `dtype` of the images returned by the
        :class:`~crappy.camera.Camera` object, as a :obj:`str`. It should
        correspond to a valid data type in :mod:`numpy`, e.g. ``'uint8'``.
        **This argument is mandatory in case** ``config`` **is** :obj:`False`.
        It is otherwise ignored.

        .. versionadded:: 2.0.0
      patch: The coordinates of the patch to track, as a :obj:`tuple` of
        exactly 4 :obj:`int`. These integers correspond to the `y` position of
        the top-left corner of the patch, the `x` position of the top-left
        corner of the patch, the height of the patch, and the width of the
        patch. Only one patch can be tracked. This argument must be provided if
        ``config`` is :obj:`False`.

        .. versionadded:: 2.0.0
      fields: The several fields to calculate on the acquired images. They
        should be given as an iterable containing :obj:`str`. Each string
        represents one field to calculate, so the more fields are given the
        heavier the computation is. The possible fields are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

        For each field, one single value is computed, corresponding to the
        norm of the field values over the patch area. If not provided, the
        default fields are ``'x', 'y', 'exx', 'eyy'``.
      labels: The labels to use for sending data to downstream Blocks. If not
        given, the default labels are
        ``'t(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)'``. They carry
        for each image its timestamp, a :obj:`dict` containing its metadata,
        and then for each field the computed value as a :obj:`float`. These
        default labels are compatible with the default fields, but must be
        changed if the number of fields changes. When setting this argument,
        make sure to give at least 2 labels for the time and the metadata, and
        one label per field. The ``'res'`` label containing the residuals if
        ``residual`` is :obj:`True` should not be included here, it will be
        automatically added.
      alpha: Weight of the smoothness term in DISFlow, as a :obj:`float`.
      delta: Weight of the color constancy term in DISFlow, as a :obj:`float`.
      gamma: Weight of the gradient constancy term in DISFlow , as a
        :obj:`float`.
      finest_scale: Finest level of the Gaussian pyramid on which the flow is
        computed in DISFlow, as an :obj:`int`. Zero level corresponds to the
        original image resolution. The final flow is obtained by bilinear
        upscaling.
      iterations: The number of fixed point iterations of variational
        refinement per scale in DISFlow, as an :obj:`int`. Set to zero to
        disable variational refinement completely. Higher values will typically
        result in more smooth and high-quality flow.
      gradient_iterations: The maximum number of gradient descent iterations in
        the patch inverse search stage in DISFlow, as an :obj:`int`. Higher
        values may improve the quality.

        .. versionchanged:: 1.5.10
          renamed from *gditerations* to *gradient_iterations*
      init: If :obj:`True`, the last calculated optical flow is used for
        initializing the calculation of the next one.
      patch_size: The size of an image patch for matching in DISFlow, in pixels
        as an :obj:`int`.
      patch_stride: The stride between two neighbor patches in DISFlow, in
        pixels as an :obj:`int`. Must be less than the ``patch_size``. Lower
        values correspond to higher flow quality.
      residual: If :obj:`True`, the residuals of the optical flow calculation
        are computed for each image. They are then returned under the ``'res'``
        label, that should not be included in the given labels. This option is
        mainly intended as a debug feature, to monitor the quality of the
        image correlation.
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.

    .. versionadded:: 1.5.10 *img_name* argument
    .. versionremoved:: 1.5.10 *residual_full* argument
    .. versionremoved:: 2.0.0 *img_name* argument
    """

    if not config and patch is None:
      raise ValueError("If the config window is disabled, the patch must be "
                       "provided !")

    super().__init__(camera=camera,
                     transform=transform,
                     config=config,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     display_freq=display_freq,
                     freq=freq,
                     debug=debug,
                     save_images=save_images,
                     img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     img_shape=img_shape,
                     img_dtype=img_dtype,
                     **kwargs)

    # Forcing the fields into a list
    if fields is None:
      fields = ["x", "y", "exx", "eyy"]
    elif isinstance(fields, str):
      fields = [fields]
    else:
      fields = list(fields)

    # Forcing the labels into a list
    if labels is None:
      self.labels = ['t(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)']
    elif isinstance(labels, str):
      self.labels = [labels]
    else:
      self.labels = list(labels)

    # Adding the residuals if required
    if residual and labels is None:
      self.labels.append('res')

    self._patch_int = patch
    self._patch: Optional[Box] = None

    # Making sure a coherent number of labels and fields was given
    if 2 + len(fields) + int(residual) != len(self.labels):
      raise ValueError(
        "The number of fields is inconsistent with the number "
        "of labels !\nMake sure that the time label was given")

    # These arguments are for the DISCorrelProcess
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
    self._residual = residual

  def prepare(self) -> None:
    """This method mostly calls the :meth:`~crappy.blocks.Camera.prepare`
    method of the parent class.

    In addition to that it instantiates the
    :class:`~crappy.blocks.camera_processes.DISCorrelProcess` object that
    performs the image correlation and the tracking.

    .. versionchanged:: 1.5.5 now accepting args and kwargs
    .. versionchanged:: 1.5.10 not accepting arguments anymore
    """

    # Instantiating the Box containing the patch to track
    if self._patch_int is not None:
      self._patch = Box(x_start=self._patch_int[1],
                        x_end=self._patch_int[1] + self._patch_int[3],
                        y_start=self._patch_int[0],
                        y_end=self._patch_int[0] + self._patch_int[2])
    else:
      self._patch = Box()

    # Instantiating the DISCorrelProcess
    self.process_proc = DISCorrelProcess(
        patch=self._patch,
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
        residual=self._residual)

    super().prepare()

  def _configure(self) -> None:
    """This method should instantiate and start the
    :class:`~crappy.tool.camera_config.DISCorrelConfig` window for configuring
    the :class:`~crappy.camera.Camera` object.

    It should also handle the case when an exception is raised in the
    configuration window.
    """

    config = None

    # Instantiating and starting the configuration window
    try:
      config = DISCorrelConfig(self._camera, self._log_queue, self._log_level,
                               self.freq, self._patch)
      config.main()

    # If an exception is raised in the config window, closing it before raising
    except (Exception,) as exc:
      self._logger.exception("Caught exception in the configuration window !",
                             exc_info=exc)
      if config is not None:
        config.stop()
      raise CameraConfigError

    # Getting the image dtype and shape for setting the shared Array
    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = config.dtype

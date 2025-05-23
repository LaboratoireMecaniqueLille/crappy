# coding: utf-8

from typing import Optional, Union, Literal
from collections.abc import Callable, Iterable
import numpy as np
from pathlib import Path

from .camera_processes import GPUCorrelProcess
from .camera import Camera

field_type = Literal['x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z']


class GPUCorrel(Camera):
  """This Block can perform GPU-accelerated image correlation on images
  acquired by a :class:`~crappy.camera.Camera` object, and project the result
  on various fields.

  It is mostly used for computing the displacement and the strain over the
  given patch, but other fields are also available. refer to the ``fields`` and
  ``labels`` arguments for more details.

  This Block takes no input :class:`~crappy.links.Link` in a majority of
  situations, and outputs the results of image correlation. It is a subclass of
  the :class:`~crappy.blocks.Camera` Block, and inherits of all its features.
  That includes the possibility to record and to display images in real-time,
  simultaneously to the image acquisition and processing. Refer to the
  documentation of the Camera Block for more information on these features.

  This Block is very similar to the :class:`DISCorrel` Block, except this
  latter uses DISFlow to perform the image correlation and is not
  GPU-accelerated. The :class:`~crappy.blocks.GPUVE` Block also relies on
  GPU-accelerated image correlation for computing the displacement and strain
  on images, but it tracks multiple patches and uses video-extensometry.

  No Region Of Interest can be specified to this Block, so by default the
  correlation is performed on the entire image. It is however possible to set
  a mask, so that only part of the image is considered when running the
  correlation.

  Warning:
    This Block cannot run with CUDA versions greater than 11.3 ! This is due to
    a deprecation in pycuda, and is unlikely to be fixed anytime soon in Crappy
    or pycuda.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               camera: str,
               fields: Union[field_type, Iterable[field_type]],
               img_shape: tuple[int, int],
               img_dtype: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               display_images: bool = False,
               displayer_backend: Optional[Literal['cv2', 'mpl']] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: int = 0,
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
               labels: Optional[Union[str, Iterable[str]]] = None,
               discard_limit: float = 3,
               discard_ref: int = 5,
               img_ref: Optional[np.ndarray] = None,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               mask: Optional[np.ndarray] = None,
               mul: float = 3,
               res: bool = False,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      camera: The name of the :class:`~crappy.camera.Camera` object to use for
        acquiring the images. Arguments can be passed to this Camera as
        ``kwargs`` of this Block. This argument is ignored if the
        ``image_generator`` argument is provided.
      fields: The several fields to calculate on the acquired images. They
        should be given as an iterable containing :obj:`str`. Each string
        represents one field to calculate, so the more fields are given the
        heavier the computation is. The possible fields are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

        For each field, one single value is computed, corresponding to the
        norm of the field values over the patch area.
      img_shape: The shape of the images returned by the
        :class:`~crappy.camera.Camera` object as a :obj:`tuple` of :obj:`int`.
        It should correspond to the value returned by :obj:`numpy.shape`.
      img_dtype: The `dtype` of the images returned by the
        :class:`~crappy.camera.Camera` object, as a :obj:`str`. It should
        correspond to a valid data type in :mod:`numpy`, e.g. ``'uint8'``.
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output. Allows applying a post-processing
        operation to the acquired images. This is done right after the
        acquisition, so the original image is permanently lost and only the
        transformed image is displayed and/or saved and/or further processed.
        The transform operation is not parallelized, so it might negatively
        affect the acquisition framerate if it is too heavy.

        .. versionadded:: 1.5.10
      display_images: If :obj:`True`, displays the acquired images in a
        dedicated window, using the backend given in ``displayer_backend`` and
        at the frequency specified in ``displayer_framerate``. This option
        should be considered as a debug or basic follow-up feature, it is not
        intended to be very fast nor to display high-quality images. The
        maximum resolution of the displayed images in `640x480`, the images
        might be downscaled to fit in this format.

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
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is displayed, and at level `3` so much information
        is displayed that it slows the code down. This argument allows to
        adjust the precision of the log messages, while the ``debug`` argument
        is for enabling or disabling logging.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.

        .. versionadded:: 1.5.10
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0
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
      labels: The labels to use for sending data to downstream Blocks. If not
        given, the default labels are ``'t(s)', 'meta'`` followed by the names
        of the given ``fields``. They carry for each image its timestamp, a
        :obj:`dict` containing its metadata, and then for each field the
        computed value as a :obj:`float`. When setting this argument, make sure
        to give at least 2 labels for the time and the metadata, and one label
        per field. The ``'res'`` label containing the residuals if ``res`` is
        :obj:`True` should not be included here, it will be automatically
        added.
      discard_limit: If ``res`` is :obj:`True`, the result of the
        correlation is not sent to the downstream Blocks if the residuals for
        the current image are greater than ``discard_limit`` times the average
        residual for the last ``discard_ref`` images.

        .. versionchanged:: 1.5.10
          renamed from *discard_lim* to *discard_limit*
      discard_ref: If ``res`` is :obj:`True`, the result of the
        correlation is not sent to the downstream Blocks if the residuals for
        the current image are greater than ``discard_limit`` times the average
        residual for the last ``discard_ref`` images.
      img_ref: A reference image, as a 2D :obj:`numpy.array` with `dtype`
        `float32`. If given, it will be set early on the correlation class and
        the test can start from the first acquired frame. If not given, the
        first acquired image will be set as the reference, which will slow down
        the beginning of the test.

        .. versionchanged:: 1.5.10 renamed from *imgref* to *img_ref*
      levels: Number of levels of the pyramid. More levels may help converging
        on images with large strain, but may fail on images that don't contain
        low spatial frequency. Fewer levels mean that the program runs faster.

        .. versionadded:: 1.5.10
      resampling_factor: The factor by which the resolution is divided between
        each stage of the pyramid. A low factor ensures coherence between the
        stages, but is more computationally intensive. A high factor allows
        reaching a finer detail level, but may lead to a coherence loss between
        the stages.

        .. versionadded:: 1.5.10
      kernel_file: The path to the file containing the kernels to use for the
        correlation. Can be a :obj:`pathlib.Path` object or a :obj:`str`. If
        not provided, the default :ref:`GPU Kernels` are used.

        .. versionadded:: 1.5.10
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing.

        .. versionadded:: 1.5.10
      mask: The mask used for weighting the region of interest on the image. It
        is generally used to prevent unexpected behavior on the border of the
        image. Also allows to select a sub-region of the image if the
        correlation should not be performed on the entire image, as this Block
        does not accept a `patch` argument.

        .. versionadded:: 1.5.10
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk to go past the solution and to diverge. If it's too
        low, the convergence will be slower and require more iterations. `3`
        was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast.

        .. versionadded:: 1.5.10
      res: If :obj:`True`, calculates the residuals after performing the
        correlation and returns the residuals along with the correlation data.
        The residuals are always returned under the label ``'res'``, and this
        label should not be included in the ``labels`` argument.

        .. versionadded:: 1.5.10
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.

    .. versionremoved:: 1.5.10
       *fps_label*, *ext*, *input_label*, *config* and *cam_kwargs* arguments
    .. versionremoved:: 2.0.0 *img_name* argument
    """

    super().__init__(camera=camera,
                     transform=transform,
                     config=False,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     display_freq=bool(verbose),
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
    if isinstance(fields, str):
      fields = [fields]
    else:
      fields = list(fields)

    # Forcing the labels into a list
    if labels is None:
      self.labels = ['t(s)', 'meta'] + fields
    elif isinstance(labels, str):
      self.labels = [labels]
    else:
      self.labels = list(labels)

    if res:
      self.labels.append('res')
    self._calc_res = res

    # Checking that the number of fields and the number of labels match
    if 2 + len(fields) + int(res) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time label was given")

    # These arguments are for the GPUCorrelProcess
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._img_ref = img_ref
    self._verbose = verbose
    self._levels = levels
    self._resampling_factor = resampling_factor
    self._kernel_file = kernel_file
    self._iterations = iterations
    self._fields = fields
    self._mask = mask
    self._mul = mul

  def prepare(self) -> None:
    """This method mostly calls the :meth:`~crappy.blocks.Camera.prepare`
    method of the parent class.

    In addition to that it instantiates the
    :class:`~crappy.blocks.camera_processes.GPUCorrelProcess` object that
    performs the GPU-accelerated image correlation.
    
    .. versionchanged:: 1.5.5 now accepting args and kwargs
    .. versionchanged:: 1.5.10 not accepting arguments anymore
    """

    # Instantiating the GPUCorrelProcess
    self.process_proc = GPUCorrelProcess(
        discard_limit=self._discard_limit,
        discard_ref=self._discard_ref,
        calc_res=self._calc_res,
        img_ref=self._img_ref,
        verbose=self._verbose,
        levels=self._levels,
        resampling_factor=self._resampling_factor,
        kernel_file=self._kernel_file,
        iterations=self._iterations,
        fields=self._fields,
        mask=self._mask,
        mul=self._mul)

    super().prepare()

  def _configure(self) -> None:
    """No configuration window is available for this Block, so this method was
    left blank."""

    ...

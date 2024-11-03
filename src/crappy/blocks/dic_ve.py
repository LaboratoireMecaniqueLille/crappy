# coding: utf-8

from typing import Optional, Union, Literal
from collections.abc import Iterable, Callable
import numpy as np
from pathlib import Path

from .camera_processes import DICVEProcess
from .camera import Camera
from ..tool.camera_config import DICVEConfig, SpotsBoxes
from .._global import CameraConfigError


class DICVE(Camera):
  """This Block can perform video-extensometry on images acquired by a
  :class:`~crappy.camera.Camera` object, by tracking patches using Digital
  Image Correlation techniques.

  It takes no input :class:`~crappy.links.Link` in a majority of situations,
  and outputs the results of the video-extensometry. It is a subclass of the
  :class:`~crappy.blocks.Camera` Block, and inherits of all its features. That
  includes the possibility to record and to display images in real-time,
  simultaneously to the image acquisition and processing. Refer to the
  documentation of the Camera Block for more information on these features.

  This Block is quite similar to the :class:`~crappy.blocks.VideoExtenso`
  Block, except this latter tracks spots instead of patches with a texture.
  Both Blocks output similar information, although the default labels and the
  data format are slightly different. The :class:`~crappy.blocks.DISCorrel`
  Block also relies on image correlation techniques for estimating the
  strain and the displacement on acquired images, but it only performs
  correlation on a single patch and is designed to have a much greater
  accuracy on this single patch. The :class:`~crappy.blocks.GPUVE` Block also
  performs video-extensometry based on digital image correlation, but the
  correlation is GPU-accelerated. The algorithm used for the correlation is
  also different from the ones available in this Block.

  For tracking the provided patches, several image correlation techniques are
  available. The most effective one is DISFlow, for which many parameters can
  be tuned. The other techniques are lighter on the CPU but also less precise.
  For each image, several values are computed and sent to the downstream
  Blocks. See the ``labels`` argument for a complete list.

  Similar to the :class:`~crappy.tool.camera_config.CameraConfig` window that
  can be displayed by the Camera Block, this Block can display a
  :class:`~crappy.tool.camera_config.DICVEConfig` window before the test
  starts. Here, the user can also select the patches to track if they were not
  already specified as an argument.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *DISVE* to *DICVE*
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
               patches: Optional[Iterable[tuple[int, int, int, int]]] = None,
               labels: Optional[Union[str, Iterable[str]]] = None,
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
        :class:`~crappy.tool.camera_config.DICVEConfig` window is displayed
        before the test starts. There, the user can interactively adjust the
        different
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
        available for the selected :class:`~crappy.camera.Camera`, visualize
        the acquired images, and select the patches to track if they haven't
        been given in the ``patches`` argument. The test starts when closing
        the configuration window. If not enabled, the ``img_dtype``,
        ``img_shape`` and ``patches`` arguments must be provided.

        .. versionadded:: 1.5.10
      display_images: If :obj:`True`, displays the acquired images in a
        dedicated window, using the backend given in ``displayer_backend`` and
        at the frequency specified in ``displayer_framerate``. This option
        should be considered as a debug or basic follow-up feature, it is not
        intended to be very fast nor to display high-quality images. The
        maximum resolution of the displayed images in `640x480`, the images
        might be downscaled to fit in this format. In addition to the acquired
        frames, the tracked patches are also displayed on the image as an
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
      patches: The coordinates of the several patches to track, as an iterable
        (like a :obj:`list` or a :obj:`tuple`) containing one or several
        :obj:`tuple` of exactly :obj:`int` values. These integers correspond to
        the `y` position of the top-left corner of the patch, the `x` position
        of the top-left corner of the patch, the height of the patch, and the
        width of the patch. Up to 4 patches can be given and tracked. This
        argument must be provided if ``config`` is :obj:`False`.
      labels: The labels to use for sending data to downstream Blocks. If not
        given, the default labels are
        ``'t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'Disp(px)'``. They
        carry for each image its timestamp, a :obj:`dict` containing its
        metadata, a :obj:`list` containing for each patch the coordinates of
        its center in a :obj:`tuple` of :obj:`int`, the `y` and `x` strain
        values calculated from the displacement and the initial position of the
        patches, and finally a :obj:`list` containing for each patch its
        displacement in the `y` and `x` direction in a :obj:`tuple` of
        :obj:`int`. If different labels are desired, they should all be
        provided at once in an iterable of :obj:`str` containing the correct
        number of labels (6).
      method: The method to use for performing the digital image correlation.
        Should be one of :
        ::

          'Disflow', 'Pixel precision', 'Parabola', 'Lucas Kanade'

        ``'Disflow'`` uses OpenCV's DISOpticalFlow and ``'Lucas Kanade'`` uses
        OpenCV's calcOpticalFlowPyrLK, while all other methods are based on a
        basic cross-correlation in the Fourier domain. ``'Pixel precision'``
        calculates the displacement by getting the position of the maximum of
        the cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. ``'Parabola'`` refines the result of
        ``'Pixel precision'`` by interpolating the neighborhood of the maximum,
        and has thus a sub-pixel resolution.

        .. versionadded:: 1.5.9
      alpha: Weight of the smoothness term in DISFlow, as a :obj:`float`.
        Ignored if ``method`` is not ``'Disflow'``.
      delta: Weight of the color constancy term in DISFlow, as a :obj:`float`.
        Ignored if ``method`` is not ``'Disflow'``.
      gamma: Weight of the gradient constancy term in DISFlow , as a
        :obj:`float`. Ignored if ``method`` is not ``'Disflow'``.
      finest_scale: Finest level of the Gaussian pyramid on which the flow is 
        computed in DISFlow, as an :obj:`int`. Zero level corresponds to the 
        original image resolution. The final flow is obtained by bilinear 
        upscaling. Ignored if ``method`` is not ``'Disflow'``.
      iterations: The number of fixed point iterations of variational 
        refinement per scale in DISFlow, as an :obj:`int`. Set to zero to 
        disable variational refinement completely. Higher values will typically 
        result in more smooth and high-quality flow. Ignored if ``method`` is
        not ``'Disflow'``.
      gradient_iterations: The maximum number of gradient descent iterations in
        the patch inverse search stage in DISFlow, as an :obj:`int`. Higher
        values may improve the quality. Ignored if ``method`` is not
        ``'Disflow'``.

        .. versionchanged:: 1.5.10
          renamed from *gditerations* to *gradient_iterations*
      patch_size: The size of an image patch for matching in DISFlow, in pixels
        as an :obj:`int`. Ignored if ``method`` is not ``'Disflow'``.
      patch_stride: The stride between two neighbor patches in DISFlow, in
        pixels as an :obj:`int`. Must be less than the ``patch_size``. Lower
        values correspond to higher flow quality. Ignored if ``method`` is not
        ``'Disflow'``.
      border: The ratio of the patch that is kept for calculating the
        displacement, if ``method`` is ``'Disflow'``. For example if a value
        of `0.2` is given, only the center `80%` of the image is used for
        calculating the average displacement, in both directions. Ignored if
        ``method`` is not ``'Disflow'``.
      safe: If :obj:`True`, checks at each new image if the patches are not
        exiting the frame. Otherwise, the patches might exit the image which
        can lead to an unexpected behavior without raising an error.

        .. versionadded:: 1.5.7
      follow: If :obj:`True`, the position of each patch on the images is
        adjusted at each new image based on the previous computed displacement
        of this patch. If a displacement of 1 in the `x` direction was
        calculated on the previous image, and the patch is located at position
        `(x0, y0)`, the patch will be moved to position `(x0 + 1, y0)` for the
        next image. It "follows" the texture to track. Recommended if the
        expected displacement in pixels is big compared to the patch size. The
        only downside is that the patches may exit the frame if something goes
        wrong with the tracking.

        .. versionadded:: 1.5.7
      raise_on_patch_exit: If :obj:`True`, raises an exception when a tracked
        patch exits the border of the image, which stops the entire test.
        Otherwise, just logs a warning message and sleeps until the test is
        stopped in another way.

        .. versionadded:: 2.0.0
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.

    .. versionremoved:: 1.5.9 *fields* argument
    .. versionremoved:: 2.0.0 *img_name* argument
    """

    if not config and patches is None:
      raise ValueError("If the config window is disabled, patches must be "
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

    # Forcing the labels into a list
    if labels is None:
      self.labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)',
                     'Exx(%)', 'Disp(px)']
    elif isinstance(labels, str):
      self.labels = [labels]
    else:
      self.labels = list(labels)

    # Making sure a coherent number of labels and fields was given
    if len(self.labels) != 6:
      raise ValueError("The number of labels should be 6 !\n"
                       "Make sure that the time label was given")

    self._patches: Optional[SpotsBoxes] = None

    self._raise_on_exit = raise_on_patch_exit
    self._patches_int = list(patches) if patches is not None else None

    # These arguments are for the DICVEProcess
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

  def prepare(self) -> None:
    """This method mostly calls the :meth:`~crappy.blocks.Camera.prepare` 
    method of the parent class.
    
    In addition to that it instantiates the
    :class:`~crappy.blocks.camera_processes.DICVEProcess` object that performs
    the image correlation and the tracking.
    
    .. versionchanged:: 1.5.5 now accepting args and kwargs
    .. versionchanged:: 1.5.10 not accepting arguments anymore
    """

    # Instantiating the SpotsBoxes containing the patches to track
    self._patches = SpotsBoxes()
    if self._patches_int is not None:
      self._patches.set_spots(self._patches_int)
      self._patches.save_length()

    # Instantiating the DICVEProcess
    self.process_proc = DICVEProcess(
        patches=self._patches,
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
        follow=self._follow,
        raise_on_exit=self._raise_on_exit)

    super().prepare()

  def _configure(self) -> None:
    """This method should instantiate and start the
    :class:`~crappy.tool.camera_config.DICVEConfig` window for configuring the
    :class:`~crappy.camera.Camera` object.

    It should also handle the case when an exception is raised in the
    configuration window.
    """

    config = None

    # Instantiating and starting the configuration window
    try:
      config = DICVEConfig(self._camera, self._log_queue, self._log_level,
                           self.freq, self._patches)
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

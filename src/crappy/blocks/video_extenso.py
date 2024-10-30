# coding: utf-8

from typing import Optional, Union, Literal
from collections.abc import Callable, Iterable
import numpy as np
from pathlib import Path

from .camera_processes import VideoExtensoProcess
from .camera import Camera
from ..tool.camera_config import VideoExtensoConfig, SpotsDetector
from .._global import CameraConfigError


class VideoExtenso(Camera):
  """This Block can perform video-extensometry on images acquired by a
  :class:`~crappy.camera.Camera` object, by tracking spots on the images.

  It takes no input :class:`~crappy.links.Link` in a majority of situations,
  and outputs the results of the video-extensometry. It is a subclass of the
  :class:`~crappy.blocks.Camera` Block, and inherits of all its features. That
  includes the possibility to record and to display images in real-time,
  simultaneously to the image acquisition and processing. Refer to the
  documentation of the Camera Block for more information on these features.

  This Block is quite similar to the :class:`~crappy.blocks.DICVE` Block,
  except this latter tracks patches with a texture instead of spots. Both
  Blocks output similar information, although the default labels and the data
  format are slightly different. See the ``labels`` argument for more detail on
  the output values. Similar to the DICVE, the :class:`~crappy.blocks.GPUVE`
  Block also performs video-extensometry based on GPU-accelerated image
  correlation.

  Similar to the :class:`~crappy.tool.camera_config.CameraConfig` window that
  can be displayed by the Camera Block, this Block can display a
  :class:`~crappy.tool.camera_config.VideoExtensoConfig` window before the test
  starts. Here, the user can also detect and select the spots to track. It is
  currently not possible to specify the coordinates of the spots to track as an
  argument, so the use of the configuration window is mandatory. This might
  change in the future.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from Video_extenso to VideoExtenso
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
               labels: Optional[Union[str, Iterable[str]]] = None,
               raise_on_lost_spot: bool = True,
               white_spots: bool = False,
               update_thresh: bool = False,
               num_spots: Optional[int] = None,
               safe_mode: bool = False,
               border: int = 5,
               min_area: int = 150,
               blur: Optional[int] = 5,
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
        :class:`~crappy.tool.camera_config.VideoExtensoConfig` window is
        displayed before the test starts. There, the user can interactively
        adjust the different
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
        available for the selected :class:`~crappy.camera.Camera`, visualize
        the acquired images, and detect and select the spots to track. The test
        starts when closing the configuration window. **It is currently not
        possible to set this argument to** :obj:`False` **!** This might change
        in the future.

        .. versionadded:: 1.5.10
      display_images: If :obj:`True`, displays the acquired images in a
        dedicated window, using the backend given in ``displayer_backend`` and
        at the frequency specified in ``displayer_framerate``. This option
        should be considered as a debug or basic follow-up feature, it is not
        intended to be very fast nor to display high-quality images. The
        maximum resolution of the displayed images in `640x480`, the images
        might be downscaled to fit in this format. In addition to the acquired
        frames, the tracked spots are also displayed on the image as an
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
        **This argument is always ignored as** ``config`` **cannot be set to**
        :obj:`False`. This might change in the future.

        .. versionadded:: 2.0.0
      img_dtype: The `dtype` of the images returned by the
        :class:`~crappy.camera.Camera` object, as a :obj:`str`. It should
        correspond to a valid data type in :mod:`numpy`, e.g. ``'uint8'``.
        **This argument is always ignored as** ``config`` **cannot be set to**
        :obj:`False`. This might change in the future.

        .. versionadded:: 2.0.0
      labels: The labels to use for sending data to downstream Blocks. If not
        given, the default labels are
        ``'t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)'``. They carry for
        each image its timestamp, a :obj:`dict` containing its metadata, a
        :obj:`list` containing for each spot the coordinates of its center in a
        :obj:`tuple` of :obj:`int`, and the `y` and `x` strain values
        calculated from the displacement and the initial position of the
        spots. If different labels are desired, they should all be provided at
        once in an iterable of :obj:`str` containing the correct number of
        labels (5).
      raise_on_lost_spot: If :obj:`True`, raises an exception when losing the
        spots to track, which stops the test. Otherwise, stops the tracking but
        lets the test go on and silently sleeps.

        .. versionchanged:: 1.5.10 renamed from *end* to *raise_on_lost_spot*
      white_spots: If :obj:`True`, detects white objects over a black
        background, else black objects over a white background.
      update_thresh: If :obj:`True`, the grey level threshold for detecting the
        spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost.
      num_spots: The number of spots to detect, as an :obj:`int` between `1`
        and `4`. If given, will try to detect exactly that number of spots and
        will fail if not enough spots can be detected. If left to :obj:`None`,
        will detect up to `4` spots, but potentially fewer.
      safe_mode: If :obj:`True`, the Block will stop and raise an exception as
        soon as overlapping spots are detected. Otherwise, it will first try to
        reduce the detection window to get rid of overlapping. This argument
        should be used when inconsistency in the results may have critical
        consequences.
      border: When searching for the new position of a spot, the Block will
        search in the last known bounding box of this spot plus a few
        additional pixels in each direction. This argument sets the number of
        additional pixels to use. It should be greater than the expected
        "speed" of the spots, in pixels / frame. But if it's set too high,
        noise or other spots might hinder the detection.
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of the camera
        and the size of the spots to detect.
      blur: The size in pixels (as an odd :obj:`int` greater than `1`) of the
        kernel to use when applying a median blur filter to the image before
        the spot detection. If not given, no blurring is performed. A slight
        blur improves the spot detection by smoothening the noise, but also
        takes a bit more time compared to no blurring.
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.

    .. versionremoved:: 1.5.10 
       *ext*, *fps_label*, *wait_l0* and *input_label* arguments
    .. versionremoved:: 2.0.0 *img_name* argument
    """

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
      self.labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    elif isinstance(labels, str):
      self.labels = [labels]
    else:
      self.labels = list(labels)

    # Making sure a coherent number of labels was given
    if len(self.labels) != 5:
      raise ValueError("The number of labels should be 5 !\n"
                       "Make sure that the time label was given")

    self._raise_on_lost_spot = raise_on_lost_spot
    self._spot_detector = SpotsDetector()

    # These arguments are for the SpotsDetector
    self._white_spots = white_spots
    self._num_spots = num_spots
    self._min_area = min_area
    self._blur = blur
    self._update_thresh = update_thresh
    self._safe_mode = safe_mode
    self._border = border

  def prepare(self) -> None:
    """This method mostly calls the :meth:`~crappy.blocks.Camera.prepare`
    method of the parent class.

    In addition to that it instantiates the
    :class:`~crappy.blocks.camera_processes.VideoExtensoProcess` object that
    performs the video-extensometry and the tracking.
    
    .. versionchanged:: 1.5.5 now accepting args and kwargs
    .. versionchanged:: 1.5.10 not accepting arguments anymore
    """

    # Instantiating the SpotsDetector containing the spots to track
    self._spot_detector = SpotsDetector(white_spots=self._white_spots,
                                        num_spots=self._num_spots,
                                        min_area=self._min_area,
                                        blur=self._blur,
                                        update_thresh=self._update_thresh,
                                        safe_mode=self._safe_mode,
                                        border=self._border)

    # Instantiating the VideoExtensoProcess
    self.process_proc = VideoExtensoProcess(
      detector=self._spot_detector,
      raise_on_lost_spot=self._raise_on_lost_spot)

    super().prepare()

  def _configure(self) -> None:
    """This method should instantiate and start the
    :class:`~crappy.tool.camera_config.VideoExtensoConfig` window for
    configuring the :class:`~crappy.camera.Camera` object.

    It should also handle the case when an exception is raised in the
    configuration window.
    """

    config = None

    # Instantiating and starting the configuration window
    try:
      config = VideoExtensoConfig(self._camera, self._log_queue,
                                  self._log_level, self.freq,
                                  self._spot_detector)
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

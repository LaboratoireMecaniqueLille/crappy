# coding: utf-8

from typing import Union, Optional, Literal
from collections.abc import Callable
from pathlib import Path
import numpy as np
from time import time, sleep, strftime, gmtime
from types import MethodType
from multiprocessing import Array, Manager, Event, RLock, Pipe, Barrier
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import managers, synchronize, connection
from threading import BrokenBarrierError
import logging

from .meta_block import Block
from .camera_processes import Displayer, ImageSaver, CameraProcess
from ..camera import camera_dict, Camera as BaseCam, deprecated_cameras
from ..tool.camera_config import CameraConfig
from .._global import CameraPrepareError, CameraRuntimeError, CameraConfigError


class Camera(Block):
  """This Block can drive a :class:`~crappy.camera.Camera` object. It can 
  acquire images, display them and record them. It can only drive one Camera at
  once.
  
  It takes no input :class:`~crappy.links.Link` in a majority of situations,
  and usually doesn't have output Links neither. The only situations when it
  can accept input Links is when an ``image_generator`` is defined, or when
  defining a ``software_trig_label``. If ``save_images`` is set to :obj:`True`,
  and if an output Link is present, a message is sent to downstream Blocks at
  each saved image, containing the timestamp, the index, and the metadata of
  the image. They are respectively carried by the `'t(s)'`, `'img_index'` and
  `'meta'` labels. This is useful for performing an action conditionally at
  each new saved image.

  Most of the time, this Block is used for recording to the desired location
  the images it acquires. Optionally, the images can also be displayed in a
  dedicated window. Both of these features are however optional, and it is
  possible to acquire images and not do anything with them. Several options are
  available for tuning the record and the display.
  
  Before a test starts, this Block can also display a 
  :class:`~crappy.tool.camera_config.CameraConfig` window in which the user can
  visualize the acquired images, and interactively tune all the 
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` available
  for the instantiated :class:`~crappy.camera.Camera`. 
  
  Internally, this Block is only in charge of the image acquisition, and the 
  other tasks are parallelized and delegated to 
  :class:`~crappy.blocks.camera_processes.CameraProcess` objects. The display 
  is handled by the :class:`~crappy.blocks.camera_processes.Displayer`, and
  the recording by the :class:`~crappy.blocks.camera_processes.ImageSaver`.
  This Block manages the instantiation, the synchronisation and the
  termination of all the CameraProcess it controls.
  
  .. versionadded:: 1.4.0
  """

  cam_count = dict()

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[Literal['cv2', 'mpl']] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               display_freq: bool = False,
               debug: Optional[bool] = False,
               freq: Optional[float] = 200,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[Literal['sitk', 'pil',
                                              'cv2', 'npy']] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[Union[tuple[int, int],
                                         tuple[int, int, int]]] = None,
               img_dtype: Optional[str] = None,
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
        :class:`~crappy.tool.camera_config.CameraConfig` window is displayed
        before the test starts. There, the user can interactively adjust the 
        different 
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` 
        available for the selected :class:`~crappy.camera.Camera`, and 
        visualize the acquired images. The test starts when closing the 
        configuration window. If not enabled, the ``img_dtype`` and 
        ``img_shape`` arguments must be provided.

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
        the :meth:`loop` method of this Block. Depending on the framerate of
        the camera and the performance of the computer, it is not guaranteed 
        that all the acquired images will be recorded.

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
      **kwargs: Any additional argument will be passed to the 
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.
    
    .. versionadded:: 1.5.2 *no_loop* argument
    .. versionremoved:: 1.5.10
       *fps_label*, *ext*, *input_label* and *no_loop* arguments
    .. versionremoved:: 2.0.0 *img_name* argument
    """

    self._save_proc: Optional[ImageSaver] = None
    self._display_proc: Optional[Displayer] = None
    self.process_proc: Optional[CameraProcess] = None
    self._manager: Optional[managers.SyncManager] = None

    self._camera: Optional[BaseCam] = None

    super().__init__()

    self.display_freq = display_freq
    self.freq = freq
    self.niceness = -10
    self.debug = debug

    # Checking for deprecated names
    if camera in deprecated_cameras:
      raise NotImplementedError(
          f"The {camera} Camera was deprecated in version 2.0.0, and renamed "
          f"to {deprecated_cameras[camera]} ! Please update your code "
          f"accordingly and check the documentation for more information")

    # Checking if the requested camera exists in Crappy
    if image_generator is None:
      if camera not in camera_dict:
        possible = ', '.join(sorted(camera_dict.keys()))
        raise ValueError(f"Unknown Camera type : {camera} ! "
                         f"The possible types are : {possible}")
      self._camera_name = camera
    else:
      self._camera_name = 'Image Generator'

    # Counting the number of instantiated cameras for each type
    if self._camera_name not in Camera.cam_count:
      Camera.cam_count[self._camera_name] = 1
    else:
      Camera.cam_count[self._camera_name] += 1

    # Setting the other attributes
    self._trig_label = software_trig_label
    self._config_cam = config
    self._transform = transform
    self._image_generator = image_generator
    self._img_shape = img_shape
    self._img_dtype = img_dtype
    self._camera_kwargs = kwargs

    # The synchronization objects are initialized later
    self._img_array: Optional[SynchronizedArray] = None
    self._img: Optional[np.ndarray] = None
    self._metadata: Optional[managers.DictProxy] = None
    self._cam_barrier: Optional[synchronize.Barrier] = None
    self._stop_event_cam: Optional[synchronize.Event] = None
    self._overlay_conn_in: Optional[connection.Connection] = None
    self._overlay_conn_out: Optional[connection.Connection] = None
    self._save_lock: Optional[synchronize.RLock] = None
    self._disp_lock: Optional[synchronize.RLock] = None
    self._proc_lock: Optional[synchronize.RLock] = None

    self._loop_count = 0
    self._fps_count = 0
    self._last_cam_fps = time()

    # Instantiating the ImageSaver if requested
    self._save_images = save_images
    self._img_extension = img_extension
    self._save_folder = save_folder
    self._save_period = save_period
    self._save_backend = save_backend

    # Instantiating the Displayer window if requested
    self._display_images = display_images
    self._title = f"Displayer {camera} {Camera.cam_count[self._camera_name]}"
    self._framerate = displayer_framerate
    self._displayer_backend = displayer_backend

  def __del__(self) -> None:
    """Safety method called when deleting the Block and ensuring that all the
    instantiated :class:`~crappy.blocks.camera_processes.CameraProcess` as well 
    as the :obj:`~multiprocessing.Manager` are stopped before exiting.
    
    If they did not stop in time, just terminates them.
    """

    if self.process_proc is not None and self.process_proc.is_alive():
      self.process_proc.terminate()

    if self._save_proc is not None and self._save_proc.is_alive():
      self._save_proc.terminate()

    if self._display_proc is not None and self._display_proc.is_alive():
      self._display_proc.terminate()

    if self._manager is not None:
      self._manager.shutdown()

  def prepare(self) -> None:
    """Preparing the save folder, opening the camera and displaying the
    configuration GUI.
    
    This method calls the :meth:`crappy.camera.Camera.open` method of the
    :class:`~crappy.camera.Camera` object.
    """

    # Instantiating the synchronization objects
    self.log(logging.DEBUG, "Instantiating the multiprocessing "
                            "synchronization objects")
    self._manager = Manager()
    self._metadata = self._manager.dict()
    self._stop_event_cam = Event()
    self._overlay_conn_in, self._overlay_conn_out = Pipe()
    self._save_lock = RLock()
    self._disp_lock = RLock()
    self._proc_lock = RLock()

    # Instantiating the ImageSaver CameraProcess
    if self._save_images:
      self.log(logging.INFO, "Instantiating the saver process")
      # The ImageSaver sends a message on each saved image only if no
      # processing is performed and if there are output Links
      send_msg = self.process_proc is None and self.outputs
      self._save_proc = ImageSaver(img_extension=self._img_extension,
                                   save_folder=self._save_folder,
                                   save_period=self._save_period,
                                   save_backend=self._save_backend,
                                   send_msg=send_msg)

    # instantiating the Displayer CameraProcess
    if self._display_images:
      self.log(logging.INFO, "Instantiating the displayer process")
      self._display_proc = Displayer(title=self._title,
                                     framerate=self._framerate,
                                     backend=self._displayer_backend)

    # Creating the Barrier for the synchronization of the CameraProcesses
    n_proc = sum(int(proc is not None) for proc in (self.process_proc,
                                                    self._save_proc,
                                                    self._display_proc))
    if not n_proc:
      self.log(logging.WARNING, "The Block acquires images but does not save "
                                "them, nor display them, nor process them !")

    self._cam_barrier = Barrier(n_proc + 1)

    # Case when the images are artificially generated and not acquired
    if self._image_generator is not None:
      self.log(logging.INFO, "Setting the image generator camera")
      self._camera = BaseCam()
      self._camera.add_scale_setting('Exx', -100., 100., None, None, 0.)
      self._camera.add_scale_setting('Eyy', -100., 100., None, None, 0.)
      img = self._image_generator(0, 0)
      self._camera.add_software_roi(img.shape[1], img.shape[0])
      self._camera.set_all()

      def get_image(self_) -> (float, np.ndarray):
        """Method generating the frames using the ``image_generator`` argument 
        if one was provided."""
        
        return time(), self_.apply_soft_roi(self._image_generator(self_.Exx,
                                                                  self_.Eyy))

      self._camera.get_image = MethodType(get_image, self._camera)

    # Instantiating the Camera object for acquiring the images
    else:
      self._camera = camera_dict[self._camera_name]()
      self.log(logging.INFO, f"Opening the {self._camera_name} Camera")
      self._camera.open(**self._camera_kwargs)
      self.log(logging.INFO, f"Opened the {self._camera_name} Camera")

    # Displaying the configuration window if required
    if self._config_cam:
      self.log(logging.INFO, "Displaying the configuration window")
      self._configure()
      self.log(logging.INFO, "Camera configuration done")

    # Setting the camera to 'Hardware' trig if it's in 'Hdw after config' mode
    if self._camera.trigger_name in self._camera.settings and \
        getattr(self._camera,
                self._camera.trigger_name) == 'Hdw after config':
      self.log(logging.INFO, "Setting the trigger mode to Hardware")
      setattr(self._camera, self._camera.trigger_name, 'Hardware')

    # Ensuring a dtype and a shape were given for the image
    if self._img_dtype is None or self._img_shape is None:
      raise ValueError(f"Cannot launch the Camera processes for camera "
                       f"{self._camera_name} as the image shape and/or dtype "
                       f"wasn't specified.\n Please specify it in the args, or"
                       f" enable the configuration window.")

    # Instantiating the Array for sharing the frames with the CameraProcesses
    self.log(logging.DEBUG, "Instantiating the shared objects")
    self._img_array = Array(np.ctypeslib.as_ctypes_type(self._img_dtype),
                            int(np.prod(self._img_shape)))
    self._img = np.frombuffer(self._img_array.get_obj(),
                              dtype=self._img_dtype).reshape(self._img_shape)

    # Starting the CameraProcess for image processing if it was instantiated
    if self.process_proc is not None:
      self.log(logging.DEBUG, "Sharing the synchronization objects with the "
                              "image processing process")
      overlay_conn = (self._overlay_conn_in if self._display_proc is not None
                      else None)
      labels = self.labels if self.labels is not None else None
      self.process_proc.set_shared(array=self._img_array,
                                   data_dict=self._metadata,
                                   lock=self._proc_lock,
                                   barrier=self._cam_barrier,
                                   event=self._stop_event_cam,
                                   shape=self._img_shape,
                                   dtype=self._img_dtype,
                                   to_draw_conn=overlay_conn,
                                   outputs=self.outputs,
                                   labels=labels,
                                   log_queue=self._log_queue,
                                   log_level=self._log_level,
                                   display_freq=self.display_freq)
      self.log(logging.INFO, "Starting the image processing process")
      self.process_proc.start()

    # Starting the ImageSaver CameraProcess if it was instantiated
    if self._save_proc is not None:
      self.log(logging.DEBUG, "Sharing the synchronization objects with the "
                              "image saver process")
      self._save_proc.set_shared(array=self._img_array,
                                 data_dict=self._metadata,
                                 lock=self._save_lock,
                                 barrier=self._cam_barrier,
                                 event=self._stop_event_cam,
                                 shape=self._img_shape,
                                 dtype=self._img_dtype,
                                 to_draw_conn=None,
                                 outputs=self.outputs,
                                 labels=list(),
                                 log_queue=self._log_queue,
                                 log_level=self._log_level,
                                 display_freq=self.display_freq)
      self.log(logging.INFO, "Starting the image saver process")
      self._save_proc.start()

    # Starting the Displayer CameraProcess if it was instantiated
    if self._display_proc is not None:
      self.log(logging.DEBUG, "Sharing the synchronization objects with the "
                              "image displayer process")
      self._display_proc.set_shared(array=self._img_array,
                                    data_dict=self._metadata,
                                    lock=self._disp_lock,
                                    barrier=self._cam_barrier,
                                    event=self._stop_event_cam,
                                    shape=self._img_shape,
                                    dtype=self._img_dtype,
                                    to_draw_conn=self._overlay_conn_out,
                                    outputs=list(),
                                    labels=list(),
                                    log_queue=self._log_queue,
                                    log_level=self._log_level,
                                    display_freq=self.display_freq)
      self.log(logging.INFO, "Starting the image displayer process")
      self._display_proc.start()

  def begin(self) -> None:
    """This method waits for all the 
    :class:`~crappy.blocks.camera_processes.CameraProcess` to be ready, then
    releases them all at once to make sure they're synchronized.
    
    
    A :obj:`~multiprocessing.Barrier` is used for forcing the CameraProcesses
    to wait for each other.
    """

    try:
      self.log(logging.INFO, "Waiting for all Camera processes to be ready")
      self._cam_barrier.wait()
      self.log(logging.INFO, "All Camera processes ready now")
    except BrokenBarrierError:
      raise CameraPrepareError

    self._last_cam_fps = time()

  def loop(self) -> None:
    """This method receives data from upstream Blocks, acquires a frame from 
    the :class:`~crappy.camera.Camera` object, and transmits it to all the 
    :class:`~crappy.blocks.camera_processes.CameraProcess`.

    The image is acquired by calling the 
    :meth:`~crappy.camera.Camera.get_image` method of the Camera object. If
    only a timestamp is returned by this method, and not a complete :obj:`dict`
    of metadata, some basic metadata is generated here and transmitted to the
    CameraProcesses.
    
    This method also manages the software trigger if this option was set, 
    applies the image transformation function if one was given, and displays
    the FPS of the acquisition if required.
    """

    # Signaling all the Blocks to stop if a CameraProcess crashed
    if self._stop_event_cam.is_set():
      raise CameraRuntimeError

    # Receiving the data from upstream Blocks
    data = self.recv_last_data(fill_missing=False)

    # Waiting for the trig label if one was given
    if self._trig_label is not None and self._trig_label not in data:
      return
    elif self._trig_label is not None and self._trig_label in data:
      self.log(logging.DEBUG, "Software trigger signal received")

    # Updating the image generator if one was provided
    if self._image_generator is not None:
      if 'Exx(%)' in data:
        self.log(logging.DEBUG, f"Setting Exx to {data['Exx(%)']}")
        self._camera.Exx = data['Exx(%)']
      if 'Eyy(%)' in data:
        self.log(logging.DEBUG, f"Setting Eyy to {data['Eyy(%)']}")
        self._camera.Eyy = data['Eyy(%)']

    # Grabbing the frame from the Camera object
    if (ret := self._camera.get_image()) is None:
      return
    metadata, img = ret
 
    # Building the metadata dict if it was not provided
    if isinstance(metadata, float):
      metadata = {'t(s)': metadata,
                  'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S",
                                               gmtime(metadata)),
                  'SubsecTimeOriginal': f'{metadata % 1:.6f}',
                  'ImageUniqueID': self._loop_count}

    # Making the timestamp relative to the beginning of the test
    metadata['t(s)'] -= self.t0

    # Applying the transform function if one as provided
    if self._transform is not None:
      img = self._transform(img)

    # Copying the metadata and the acquired frame into the shared objects for 
    # transfer to the CameraProcesses
    # This is done with all the Locks acquired to avoid any conflict
    with self._save_lock, self._disp_lock, self._proc_lock:
      self.log(logging.DEBUG, f"Writing metadata to shared dict: {metadata}")
      self._metadata.update(metadata)
      self.log(logging.DEBUG, "Writing image to shared array")
      np.copyto(self._img, img)

    self._loop_count += 1

    # If requested, displays the FPS of the image acquisition
    if self.display_freq:
      self._fps_count += 1
      t = time()
      if t - self._last_cam_fps > 2:
        self.log(logging.INFO, f"Acquisition FPS: "
                               f"{self._fps_count / (t - self._last_cam_fps)}")
        self._last_cam_fps = t
        self._fps_count = 0

  def finish(self) -> None:
    """This method stops the image acquisition on the 
    :class:`~crappy.camera.Camera`, as well as all the 
    :class:`~crappy.blocks.camera_processes.CameraProcess` that were started.
    
    If the CameraProcesses do not gently stop, they are terminated. Also stops
    the :obj:`~multiprocessing.Manager` in charge of handling the metadata.
    
    For stopping the image acquisition, the :meth:`~crappy.camera.Camera.close`
    method is called.
    """

    # Closing the Camera object
    if self._image_generator is None and self._camera is not None:
      self.log(logging.INFO, f"Closing the {self._camera_name} Camera")
      self._camera.close()
      self.log(logging.INFO, f"Closed the {self._camera_name} Camera")

    # Setting the stop event to signal all CameraProcesses to stop
    if self._stop_event_cam is not None:
      self.log(logging.DEBUG, "Asking all the children processes to stop")
      self._stop_event_cam.set()
      sleep(0.2)

    # If the processing CameraProcess is not done, terminating it
    if self.process_proc is not None and self.process_proc.is_alive():
      self.log(logging.WARNING, "Image processing process not stopped, "
                                "killing it !")
      self.process_proc.terminate()
    # If the ImageSaver CameraProcess is not done, terminating it
    if self._save_proc is not None and self._save_proc.is_alive():
      self.log(logging.WARNING, "Image saver process not stopped, "
                                "killing it !")
      self._save_proc.terminate()
    # If the Displayer CameraProcess is not done, terminating it
    if self._display_proc is not None and self._display_proc.is_alive():
      self.log(logging.WARNING, "Image displayer process not stopped, "
                                "killing it !")
      self._display_proc.terminate()

    # Closing the Manager handling the metadata
    if self._manager is not None:
      self._manager.shutdown()

  def _configure(self) -> None:
    """This method should instantiate and start the 
    :class:`~crappy.tool.camera_config.CameraConfig` window for configuring the
    :class:`~crappy.camera.Camera` object.
    
    It should also handle the case when an exception is raised in the 
    configuration window.
    
    This method is meant to be overridden by children of the Camera Block, as
    other image processing Blocks rely on subclasses of 
    :class:`~crappy.tool.camera_config.CameraConfig`.
    """

    config = None
    
    # Instantiating and starting the configuration window
    try:
      config = CameraConfig(self._camera, self._log_queue,
                            self._log_level, self.freq)
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

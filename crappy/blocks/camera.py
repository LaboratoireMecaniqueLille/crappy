# coding: utf-8

from typing import Callable, Union, Optional, Tuple
from pathlib import Path
import numpy as np
from time import time, sleep, strftime, gmtime
from types import MethodType
from multiprocessing import Array, Manager, Event, RLock, Pipe, Barrier
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import managers, synchronize, connection
from threading import BrokenBarrierError
from math import prod
import logging

from .meta_block import Block
from .camera_processes import Displayer, ImageSaver, CameraProcess
from ..camera import camera_dict, Camera as BaseCam
from ..tool.camera_config import CameraConfig
from .._global import CameraPrepareError, CameraRuntimeError


class Camera(Block):
  """"""

  cam_count = dict()

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: bool = False,
               debug: bool = False,
               freq: float = 200,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[Tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
               **kwargs) -> None:
    """"""

    self._save_proc: Optional[ImageSaver] = None
    self._display_proc: Optional[Displayer] = None
    self._process_proc: Optional[CameraProcess] = None

    self._camera: Optional[BaseCam] = None

    super().__init__()

    self.verbose = verbose
    self.freq = freq
    self.niceness = -10
    self.log_level = logging.DEBUG if debug else logging.INFO

    # Checking if the requested camera exists in Crappy
    if image_generator is None:
      if camera.capitalize() not in camera_dict:
        raise ValueError(f"No camera named {camera.capitalize()} found in the "
                         f"list of available cameras !")
      self._camera_name = camera.capitalize()
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

    # The objects must be initialized later for Windows compatibility
    self._img_array: Optional[SynchronizedArray] = None
    self._img: Optional[np.ndarray] = None
    self._manager: Optional[managers.SyncManager] = None
    self._metadata: Optional[managers.DictProxy] = None
    self._cam_barrier: Optional[synchronize.Barrier] = None
    self._stop_event_cam: Optional[synchronize.Event] = None
    self._box_conn_in: Optional[connection.Connection] = None
    self._box_conn_out: Optional[connection.Connection] = None
    self._save_lock: Optional[synchronize.RLock] = None
    self._disp_lock: Optional[synchronize.RLock] = None
    self._proc_lock: Optional[synchronize.RLock] = None

    self._loop_count = 0
    self._fps_count = 0
    self._last_cam_fps = time()

    # Cannot start process from __main__
    if not save_images:
      self._save_proc_kw = None
    else:
      self._save_proc_kw = dict(img_extension=img_extension,
                                save_folder=save_folder,
                                save_period=save_period,
                                save_backend=save_backend)

    # Instantiating the displayer window if requested
    if not display_images:
      self._display_proc_kw = None
    else:
      self._display_proc_kw = dict(
        title=f"Displayer {camera} "
              f"{Camera.cam_count[self._camera_name]}",
        framerate=displayer_framerate, backend=displayer_backend)

  def __del__(self) -> None:
    """"""

    if self._process_proc is not None and self._process_proc.is_alive():
      self._process_proc.terminate()

    if self._save_proc is not None and self._save_proc.is_alive():
      self._save_proc.terminate()

    if self._display_proc is not None and self._display_proc.is_alive():
      self._display_proc.terminate()

    if self._manager is not None:
      self._manager.shutdown()

  def prepare(self) -> None:
    """Preparing the save folder, opening the camera and displaying the
    configuration GUI."""

    # Instantiating the multiprocessing objects
    self.log(logging.DEBUG, "Instantiating the multiprocessing "
                            "synchronization objects")
    self._manager = Manager()
    self._metadata = self._manager.dict()
    self._stop_event_cam = Event()
    self._box_conn_in, self._box_conn_out = Pipe()
    self._save_lock = RLock()
    self._disp_lock = RLock()
    self._proc_lock = RLock()

    if self._save_proc_kw is not None:
      self.log(logging.INFO, "Instantiating the saver process")
      self._save_proc = ImageSaver(log_queue=self._log_queue,
                                   log_level=self.log_level,
                                   verbose=self.verbose,
                                   **self._save_proc_kw)

    if self._display_proc_kw is not None:
      self.log(logging.INFO, "Instantiating the displayer process")
      self._display_proc = Displayer(log_queue=self._log_queue,
                                     log_level=self.log_level,
                                     verbose=self.verbose,
                                     **self._display_proc_kw)

    # Creating the barrier for camera processes synchronization
    n_proc = sum(int(proc is not None) for proc in (self._process_proc,
                                                    self._save_proc,
                                                    self._display_proc))
    if not n_proc:
      self.log(logging.WARNING, "The block acquires images but does not save "
                                "them, nor display them, nor process them !")

    self._cam_barrier = Barrier(n_proc + 1)

    # Case when the images are generated and not acquired
    if self._image_generator is not None:
      self.log(logging.INFO, "Setting the image generator camera")
      self._camera = BaseCam()
      self._camera.add_scale_setting('Exx', -100, 100, None, None, 0.)
      self._camera.add_scale_setting('Eyy', -100, 100, None, None, 0.)
      self._camera.set_all()

      def get_image(self_) -> (float, np.ndarray):
        return time(), self._image_generator(self_.Exx, self_.Eyy)

      self._camera.get_image = MethodType(get_image, self._camera)

    # Case when an actual camera object is responsible for acquiring the images
    else:
      self._camera = camera_dict[self._camera_name]()
      self.log(logging.INFO, f"Opening the {self._camera_name} Camera")
      self._camera.open(**self._camera_kwargs)
      self.log(logging.INFO, f"Opened the {self._camera_name} Camera")

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

    if self._img_dtype is None or self._img_shape is None:
      raise ValueError(f"Cannot launch the Camera processes for camera "
                       f"{self._camera_name} as the image shape and/or dtype "
                       f"wasn't specified.\n Please specify it in the args, or"
                       f" enable the configuration window.")

    self.log(logging.DEBUG, "Instantiating the shared objects")
    self._img_array = Array(np.ctypeslib.as_ctypes_type(self._img_dtype),
                            prod(self._img_shape))
    self._img = np.frombuffer(self._img_array.get_obj(),
                              dtype=self._img_dtype).reshape(self._img_shape)

    if self._process_proc is not None:
      self.log(logging.DEBUG, "Sharing the synchronization objects with the "
                              "image processing process")
      box_conn = self._box_conn_in if self._display_proc is not None else None
      self._process_proc.set_shared(array=self._img_array,
                                    data_dict=self._metadata,
                                    lock=self._proc_lock,
                                    barrier=self._cam_barrier,
                                    event=self._stop_event_cam,
                                    shape=self._img_shape,
                                    dtype=self._img_dtype,
                                    box_conn=box_conn,
                                    outputs=self.outputs,
                                    labels=self.labels)
      self.log(logging.INFO, "Starting the image processing process")
      self._process_proc.start()

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
                                 box_conn=None,
                                 outputs=list(),
                                 labels=list())
      self.log(logging.INFO, "Starting the image saver process")
      self._save_proc.start()

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
                                    box_conn=self._box_conn_out,
                                    outputs=list(),
                                    labels=list())
      self.log(logging.INFO, "Starting the image displayer process")
      self._display_proc.start()

  def begin(self) -> None:
    """"""

    try:
      self.log(logging.INFO, "Waiting for all Camera processes to be ready")
      self._cam_barrier.wait()
      self.log(logging.INFO, "All Camera processes ready now")
    except BrokenBarrierError:
      raise CameraPrepareError

    self._last_cam_fps = time()

  def loop(self) -> None:
    """Receives the incoming data, acquires an image, displays it, saves it,
    and finally processes it if needed."""

    if self._stop_event_cam.is_set():
      raise CameraRuntimeError

    data = self.recv_last_data(fill_missing=False)

    # Waiting for the trig label if it was given
    if self._trig_label is not None and self._trig_label not in data:
      return
    elif self._trig_label is not None and self._trig_label in data:
      self.log(logging.DEBUG, "Software trigger signal received")

    # Updating the image generator if there's one
    if self._image_generator is not None:
      if 'Exx(%)' in data:
        self.log(logging.DEBUG, f"Setting Exx to {data['Exx(%)']}")
        self._camera.Exx = data['Exx(%)']
      if 'Eyy(%)' in data:
        self.log(logging.DEBUG, f"Setting Eyy to {data['Eyy(%)']}")
        self._camera.Eyy = data['Eyy(%)']

    # Actually getting the image from the camera object
    ret = self._camera.get_image()
    if ret is None:
      return
    metadata, img = ret

    # Building the metadata if it was not provided
    if isinstance(metadata, float):
      metadata = {'t(s)': metadata,
                  'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S",
                                               gmtime(metadata)),
                  'SubsecTimeOriginal': f'{metadata % 1:.6f}',
                  'ImageUniqueID': self._loop_count}

    metadata['t(s)'] -= self.t0

    # Applying the transform function
    if self._transform is not None:
      img = self._transform(img)

    with self._save_lock, self._disp_lock, self._proc_lock:
      self.log(logging.DEBUG, f"Writing metadata to shared dict: {metadata}")
      self._metadata.update(metadata)
      self.log(logging.DEBUG, "Writing image to shared array")
      np.copyto(self._img, img)

    self._loop_count += 1

    if self.verbose:
      self._fps_count += 1
      t = time()
      if t - self._last_cam_fps > 2:
        self.log(logging.INFO, f"Acquisition FPS: "
                               f"{self._fps_count / (t - self._last_cam_fps)}")
        self._last_cam_fps = t
        self._fps_count = 0

  def finish(self) -> None:
    """"""

    if self._image_generator is None and self._camera is not None:
      self.log(logging.INFO, f"Closing the {self._camera_name} Camera")
      self._camera.close()
      self.log(logging.INFO, f"Closed the {self._camera_name} Camera")

    if self._stop_event_cam is not None:
      self.log(logging.DEBUG, "Asking all the children processes to stop")
      self._stop_event_cam.set()
      sleep(0.2)

    if self._process_proc is not None and self._process_proc.is_alive():
      self.log(logging.WARNING, "Image processing process not stopped, "
                                "killing it !")
      self._process_proc.terminate()
    if self._save_proc is not None and self._save_proc.is_alive():
      self.log(logging.WARNING, "Image saver process not stopped, "
                                "killing it !")
      self._save_proc.terminate()
    if self._display_proc is not None and self._display_proc.is_alive():
      self.log(logging.WARNING, "Image displayer process not stopped, "
                                "killing it !")
      self._display_proc.terminate()

    if self._manager is not None:
      self._manager.shutdown()

  def _configure(self) -> None:
    """"""

    config = CameraConfig(self._camera)
    config.main()
    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = config.dtype

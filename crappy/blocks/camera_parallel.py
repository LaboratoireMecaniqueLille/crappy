# coding: utf-8

from typing import Callable, Union, Optional, Tuple
from pathlib import Path
import numpy as np
from time import time, sleep, strftime, gmtime
from types import MethodType
from multiprocessing import Array, Manager, Event, RLock, Pipe
from multiprocessing.sharedctypes import SynchronizedArray
from math import prod

from .block import Block
from .camera_parallel_display import Displayer
from .camera_parallel_record import Image_saver
from ..camera import camera_list, Camera as BaseCam
from ..tool import Camera_config


class Camera_parallel(Block):
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
               freq: float = 200,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[Tuple[int, ...]] = None,
               img_dtype: Optional[str] = None,
               **kwargs) -> None:
    """"""

    self._save_proc = None
    self._display_proc = None

    super().__init__()

    self.verbose = verbose
    self.freq = freq
    self.niceness = -10

    # Checking if the requested camera exists in Crappy
    if image_generator is None:
      if camera.capitalize() not in camera_list:
        raise ValueError(f"No camera named {camera.capitalize()} found in the "
                         f"list of available cameras !")
      self._camera_name = camera.capitalize()
    else:
      self._camera_name = 'Image Generator'

    # Counting the number of instantiated cameras for each type
    if self._camera_name not in Camera_parallel.cam_count:
      Camera_parallel.cam_count[self._camera_name] = 1
    else:
      Camera_parallel.cam_count[self._camera_name] += 1

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
              f"{Camera_parallel.cam_count[self._camera_name]}",
        framerate=displayer_framerate, backend=displayer_backend)

    # Setting the other attributes
    self._trig_label = software_trig_label
    self._config_cam = config
    self._transform = transform
    self._image_generator = image_generator
    self._img_shape = img_shape
    self._img_dtype = img_dtype
    self._camera_kwargs = kwargs

    self._img_array: Optional[SynchronizedArray] = None
    self._img: Optional[np.ndarray] = None
    self._manager = Manager()
    self._metadata = self._manager.dict()
    self._stop_event = Event()
    self._box_conn_in, self._box_conn_out = Pipe()
    self._save_lock = RLock()
    self._disp_lock = RLock()
    self._proc_lock = RLock()

    self._n_loops = 0

  def __del__(self) -> None:
    """"""

    if self._save_proc is not None and self._save_proc.is_alive():
      self._save_proc.terminate()

    if self._display_proc is not None and self._display_proc.is_alive():
      self._display_proc.terminate()

    self._manager.shutdown()

  def prepare(self) -> None:
    """Preparing the save folder, opening the camera and displaying the
    configuration GUI."""

    if self._save_proc_kw is not None:
      self._save_proc = Image_saver(**self._save_proc_kw)

    if self._display_proc_kw is not None:
      self._display_proc = Displayer(**self._display_proc_kw)

    # Case when the images are generated and not acquired
    if self._image_generator is not None:
      self._camera = BaseCam()
      self._camera.add_scale_setting('Exx', 0., 100., None, None, 0.)
      self._camera.add_scale_setting('Eyy', 0., 100., None, None, 0.)
      self._camera.set_all()

      def get_image(self_) -> (float, np.ndarray):
        return time(), self._image_generator(self_.Exx, self_.Eyy)

      self._camera.get_image = MethodType(get_image, self._camera)

    # Case when an actual camera object is responsible for acquiring the images
    else:
      self._camera = camera_list[self._camera_name]()
      self._camera.open(**self._camera_kwargs)

    if self._config_cam:
      config = Camera_config(self._camera)
      config.main()
      if config.shape is not None:
        self._img_shape = config.shape
      if config.dtype is not None:
        self._img_dtype = config.dtype

    # Setting the camera to 'Hardware' trig if it's in 'Hdw after config' mode
    if self._camera.trigger_name in self._camera.settings and \
        getattr(self._camera,
                self._camera.trigger_name) == 'Hdw after config':
      setattr(self._camera, self._camera.trigger_name, 'Hardware')

    if self._img_dtype is None or self._img_shape is None:
      raise ValueError(f"Cannot launch the Camera processes for camera "
                       f"{self._camera_name} as the image shape and/or dtype "
                       f"wasn't specified.\n Please specify it in the args, or"
                       f" enable the configuration window.")

    self._img_array = Array(np.ctypeslib.as_ctypes_type(self._img_dtype),
                            prod(self._img_shape))
    self._img = np.frombuffer(self._img_array.get_obj(),
                              dtype=self._img_dtype).reshape(self._img_shape)

    if self._save_proc is not None:
      self._save_proc.set_shared(array=self._img_array,
                                 data_dict=self._metadata,
                                 lock=self._save_lock,
                                 event=self._stop_event,
                                 shape=self._img_shape,
                                 dtype=self._img_dtype)
      self._save_proc.start()

    if self._display_proc is not None:
      self._display_proc.set_shared(array=self._img_array,
                                    data_dict=self._metadata,
                                    lock=self._disp_lock,
                                    event=self._stop_event,
                                    shape=self._img_shape,
                                    dtype=self._img_dtype,
                                    box_conn=self._box_conn_out)
      self._display_proc.start()

  def loop(self) -> None:
    """Receives the incoming data, acquires an image, displays it, saves it,
    and finally processes it if needed."""

    data = self.recv_all_last()

    # Waiting for the trig label if it was given
    if self._trig_label is not None and self._trig_label not in data:
      return

    # Updating the image generator if there's one
    if self._image_generator is not None:
      if 'Exx(%)' in data:
        self._camera.Exx = data['Exx(%)']
      if 'Eyy(%)' in data:
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
                  'ImageUniqueID': self._n_loops}

    metadata['t(s)'] -= self.t0

    self._n_loops += 1

    # Applying the transform function
    if self._transform is not None:
      img = self._transform(img)

    with self._save_lock, self._disp_lock, self._proc_lock:
      self._metadata.update(metadata)
      np.copyto(self._img, img)

  def finish(self) -> None:
    """"""

    if self._image_generator is None:
      self._camera.close()

    self._stop_event.set()
    sleep(0.1)
    if self._save_proc is not None and self._save_proc.is_alive():
      self._save_proc.terminate()
    if self._display_proc is not None and self._display_proc.is_alive():
      self._display_proc.terminate()

    self._manager.shutdown()

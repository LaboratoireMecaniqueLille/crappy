# coding: utf-8

from multiprocessing import Process, managers, get_start_method
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.queues import Queue
from csv import DictWriter
from time import strftime, gmtime
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path
import logging
import logging.handlers
from .._global import OptionalModule

try:
  import SimpleITK as Sitk
except (ModuleNotFoundError, ImportError):
  Sitk = OptionalModule("SimpleITK")

try:
  import PIL
  from PIL.ExifTags import TAGS
  TAGS_INV = {val: key for key, val in TAGS.items()}
except (ModuleNotFoundError, ImportError):
  PIL = OptionalModule("Pillow")
  TAGS = TAGS_INV = OptionalModule("Pillow")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Image_saver(Process):
  """"""

  def __init__(self,
               log_queue: Queue,
               parent_name: str,
               log_level: int = 20,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None) -> None:
    """"""

    super().__init__()

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._parent_name = parent_name

    # Trying the different possible backends and checking if the given one
    # is correct
    if save_backend is None:
      if not isinstance(Sitk, OptionalModule):
        self._save_backend = 'sitk'
      elif not isinstance(PIL, OptionalModule):
        self._save_backend = 'cv2'
      elif not isinstance(PIL, OptionalModule):
        self._save_backend = 'pil'
      else:
        raise ModuleNotFoundError("Neither SimpleITK, opencv-python nor "
                                  "Pillow could be imported, no backend "
                                  "found for saving the images")
    elif save_backend in ('sitk', 'pil', 'cv2'):
      self._save_backend = save_backend
    else:
      raise ValueError("The save_backend argument should be either 'sitk', "
                       "'pil' or 'cv2' !")

    self._img_extension = img_extension

    # Setting a default save folder if not given
    if save_folder is None:
      self._save_folder = Path.cwd() / 'Crappy_images'
    else:
      self._save_folder = Path(save_folder)

    self._save_period = int(save_period)

    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[Tuple[int, int]] = None

    self._metadata = {'ImageUniqueID': -float('inf')}
    self._img: Optional[np.ndarray] = None
    self._dtype = None

    self._csv_created = False
    self._csv_path = None

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
                 event: Event,
                 shape: Tuple[int, int],
                 dtype) -> None:
    """"""

    self._img_array = array
    self._data_dict = data_dict
    self._lock = lock
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype

    self._img = np.empty(shape=shape, dtype=dtype)

    # Creating the folder for saving the images if it doesn't exist
    if self._save_folder is not None:
      if not self._save_folder.exists():
        Path.mkdir(self._save_folder, exist_ok=True, parents=True)

  def run(self) -> None:
    """"""

    try:
      self._set_logger()
      self._log(logging.INFO, "Logger configured")

      while not self._stop_event.is_set():
        save = False
        with self._lock:

          if 'ImageUniqueID' not in self._data_dict:
            continue

          if self._data_dict['ImageUniqueID'] != \
              self._metadata['ImageUniqueID'] and \
              self._data_dict['ImageUniqueID'] - \
              self._metadata['ImageUniqueID'] >= self._save_period:
            self._metadata = self._data_dict.copy()
            save = True

            self._log(logging.DEBUG, f"Got new image to save with id "
                                     f"{self._metadata['ImageUniqueID']}")

            np.copyto(self._img,
                      np.frombuffer(self._img_array.get_obj(),
                                    dtype=self._dtype).reshape(self._shape))

        if save:
          self._save()

      self._log(logging.INFO, "Stop event set, stopping the recording")

    except KeyboardInterrupt:
      self._log(logging.INFO, "KeyboardInterrupt caught, stopping the "
                              "recording")

  def _save(self) -> None:
    """Simply saves the given image to the given path using the selected
    backend."""

    if not self._csv_created:
      self._csv_path = self._save_folder / \
          f'metadata_{strftime("%d_%m_%y %H:%M:%S", gmtime())}.csv'

      self._log(logging.INFO, f"Creating file for saving the metadata: "
                              f"{self._csv_path}")

      with open(self._csv_path, 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=self._metadata.keys())
        writer.writeheader()

      self._csv_created = True

    self._log(logging.DEBUG, f"Saving metadata: {self._metadata}")
    with open(self._csv_path, 'a') as csvfile:
      writer = DictWriter(csvfile, fieldnames=self._metadata.keys())
      writer.writerow({**self._metadata, 't(s)': self._metadata['t(s)']})

    path = str(self._save_folder / f"{self._metadata['ImageUniqueID']} "
                                   f"{self._metadata['t(s)']:.3f}."
                                   f"{self._img_extension}")

    self._log(logging.DEBUG, "Saving image")
    if self._save_backend == 'sitk':
      Sitk.WriteImage(Sitk.GetImageFromArray(self._img), path)

    elif self._save_backend == 'cv2':
      cv2.imwrite(path, self._img)

    elif self._save_backend == 'pil':
      PIL.Image.fromarray(self._img).save(
        path, exif={TAGS_INV[key]: val for key, val in self._metadata.items()
                    if key in TAGS_INV})

  def _set_logger(self) -> None:
    """"""

    log_level = 10 * int(round(self._log_level / 10, 0))

    logger = logging.getLogger(f'crappy.{self._parent_name}.Recorder')
    logger.setLevel(min(log_level, logging.INFO))

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn":
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(min(log_level, logging.INFO))
      logger.addHandler(queue_handler)

    self._logger = logger

  def _log(self, level: int, msg: str) -> None:
    """Sends a log message to the logger.

    Args:
      level: The logging level, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(level, msg)

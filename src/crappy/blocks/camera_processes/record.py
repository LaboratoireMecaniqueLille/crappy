# coding: utf-8

from multiprocessing.queues import Queue
from csv import DictWriter
from time import strftime, gmtime
import numpy as np
from typing import Optional, Union
from pathlib import Path
import logging
import logging.handlers

from .camera_process import CameraProcess
from ..._global import OptionalModule

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


class ImageSaver(CameraProcess):
  """"""

  def __init__(self,
               log_queue: Queue,
               log_level: int = 20,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               display_freq: bool = False) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     display_freq=display_freq)

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

    self._csv_created = False
    self._csv_path = None

  def _init(self) -> None:
    """"""

    if self._save_folder is not None and not self._save_folder.exists():
      self._log(logging.INFO, f"Creating the folder for saving images at: "
                              f"{self._save_folder}")
      Path.mkdir(self._save_folder, exist_ok=True, parents=True)

  def _get_data(self) -> bool:
    """"""

    with self._lock:

      if 'ImageUniqueID' not in self._data_dict:
        return False

      if self._data_dict['ImageUniqueID'] == self._metadata['ImageUniqueID']:
        return False

      if self._metadata['ImageUniqueID'] is not None and \
          self._data_dict['ImageUniqueID'] - self._metadata['ImageUniqueID'] \
          < self._save_period:
        return False

      self._metadata = self._data_dict.copy()

      self._log(logging.DEBUG, f"Got new image to process with id "
                               f"{self._metadata['ImageUniqueID']}")

      np.copyto(self._img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return
    self.fps_count += 1

    if not self._csv_created:
      self._csv_path = (self._save_folder /
                        f'metadata_'
                        f'{strftime("%d_%m_%y %H:%M:%S", gmtime())}.csv')

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

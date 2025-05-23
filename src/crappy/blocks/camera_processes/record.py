# coding: utf-8

from csv import DictWriter
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
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can record
  images acquired by a :class:`~crappy.blocks.Camera` Block to the desired
  location and in the desired format.

  Various backends can be used for recording the images, some may be faster or
  slower depending on the machine. It is possible to only save one out of a
  given number of images, if not all frames are needed.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               send_msg: bool = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      img_extension: The file extension for the recorded images, as a
        :obj:`str` and without the dot. Common file extensions include `tiff`,
        `png`, `jpg`, etc.
      save_folder: Path to the folder where to save the images. Can be an
        absolute or a relative path. The folder does not need to already exit,
        in which case it is created.
      save_period: Only one out of that number images at most will be saved.
        Allows to have a known periodicity in case the framerate is too high
        to record all the images. Or simply to reduce the number of saved
        images if saving them all is not needed.
      save_backend: The backend to use for saving the images. Should be one of:
        ::

          'sitk', 'pil', 'cv2', 'npy'

        They correspond to the modules :mod:`SimpleITK`, :mod:`PIL` (Pillow
        Fork), :mod:`cv2` (OpenCV), and :mod:`numpy`. Depending on the machine,
        some may be faster or slower. The ``img_extension`` is ignored for the
        backend ``'npy'``, that saves the images as raw numpy arrays.
      send_msg: In case no processing is performed, and if output Links are
        present, this argument is set to :obj:`True`. In that case, a message
        containing the timestamp, the index, and the metadata of the image is
        sent to downstream Blocks each time an image is saved.

        .. versionadded:: 2.0.5
    """

    super().__init__()

    # Trying the different possible backends and checking if the given one
    # is correct
    if save_backend is None:
      if not isinstance(Sitk, OptionalModule):
        self._save_backend = 'sitk'
      elif not isinstance(PIL, OptionalModule):
        self._save_backend = 'pil'
      elif not isinstance(cv2, OptionalModule):
        self._save_backend = 'cv2'
      else:
        self._save_backend = 'npy'
    elif save_backend in ('sitk', 'pil', 'cv2', 'npy'):
      self._save_backend = save_backend
    else:
      raise ValueError("The save_backend argument should be either 'sitk', "
                       "'pil', 'cv2' or 'npy' !")

    # In case the images are saved as arrays, don't include extension
    self._img_extension = img_extension if self._save_backend != 'npy' else ''

    # Setting a default save folder if not given
    if save_folder is None:
      self._save_folder = Path.cwd() / 'Crappy_images'
    else:
      self._save_folder = Path(save_folder)

    self._save_period = int(save_period)
    self._send_msg: bool = send_msg

    self._csv_created = False
    self._csv_path = None
    self._metadata_name = 'metadata.csv'

  def init(self) -> None:
    """Creates the folder for saving the images.

    If a folder is already present at the indicated path and contains images,
    saving to a new folder with the same name but ending with a suffix.
    """

    # If the save folder already exists, checking if it contains images by
    # checking if a metadata file is present
    if self._save_folder.exists():
      content = (path.name for path in self._save_folder.iterdir())
      # If it contains images, saving to a different folder
      if self._metadata_name in content:
        self.log(logging.WARNING, f"The folder {self._save_folder} already "
                                  f"seems to contain images from Crappy !")
        parent, name = self._save_folder.parent, self._save_folder.name
        i = 1
        # Adding an integer at the end of the folder name to differentiate it
        while (parent / f'{name}_{i:05d}').exists():
          i += 1
        self._save_folder = parent / f'{name}_{i:05d}'
        self.log(logging.WARNING, f"Saving the images at {self._save_folder} "
                                  f"instead !")

      else:
        self.log(logging.DEBUG,
                 f"The folder {self._save_folder} for recording images exists"
                 f" but does not contain images yet.")

    # Creating the folder for recording images
    if not self._save_folder.exists():
      self.log(logging.INFO, f"Creating the folder for saving images at: "
                             f"{self._save_folder}")
      Path.mkdir(self._save_folder, exist_ok=True, parents=True)

  def _get_data(self) -> bool:
    """Method similar to the one of the parent class, except it also ensures
    that at most only one out of ``save_period`` images is being saved.

    Returns:
      :obj:`True` in case a frame was acquired and needs to be handled, or
      :obj:`False` if no frame was grabbed and nothing should be done.
    """

    # Acquiring the Lock to avoid conflicts with other CameraProcesses
    with self._lock:

      # In case there's no frame grabbed yet
      if 'ImageUniqueID' not in self._data_dict:
        return False

      # In case the frame in buffer was already handled during a previous loop,
      if self._data_dict['ImageUniqueID'] == self.metadata['ImageUniqueID']:
        return False

     # In case it's too early to save the new frame
      if self.metadata['ImageUniqueID'] is not None and \
          self._data_dict['ImageUniqueID'] - self.metadata['ImageUniqueID'] \
          < self._save_period:
        return False

      # Copying the metadata
      self.metadata = self._data_dict.copy()

      self.log(logging.DEBUG, f"Got new image to process with id "
                              f"{self.metadata['ImageUniqueID']}")

      # Copying the frame
      np.copyto(self.img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def loop(self) -> None:
    """This method grabs the latest frame, writes its metadata to a `.csv` file
    and saves the image at the chosen location using the chosen backend.

    On the first frame, the metadata file is created and its header is
    populated using the metadata of the frame.
    """

    # Creating the .csv containing the metadata on the first received frame
    if not self._csv_created:
      self._csv_path = (self._save_folder / self._metadata_name)

      self.log(logging.INFO, f"Creating file for saving the metadata: "
                             f"{self._csv_path}")

      # Also writing the header of the .csv file when creating it
      with open(self._csv_path, 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=self.metadata.keys())
        writer.writeheader()

      self._csv_created = True

    # Saving the received metadata to the .csv file
    self.log(logging.DEBUG, f"Saving metadata: {self.metadata}")
    with open(self._csv_path, 'a') as csvfile:
      writer = DictWriter(csvfile, fieldnames=self.metadata.keys())
      writer.writerow({**self.metadata, 't(s)': self.metadata['t(s)']})

    # Only include the extension for the image file if applicable
    if self._img_extension:
      path = str(self._save_folder / f"{self.metadata['ImageUniqueID']:06d}_"
                                     f"{self.metadata['t(s)']:.3f}."
                                     f"{self._img_extension}")
    else:
      path = str(self._save_folder / f"{self.metadata['ImageUniqueID']:06d}_"
                                     f"{self.metadata['t(s)']:.3f}")

    # Saving the image at the destination path using the chosen backend
    self.log(logging.DEBUG, "Saving image")
    if self._save_backend == 'sitk':
      if len(self.img.shape) == 3:
        Sitk.WriteImage(Sitk.GetImageFromArray(self.img[:, :, ::-1],
                                               isVector=True), path)
      else:
        Sitk.WriteImage(Sitk.GetImageFromArray(self.img), path)

    elif self._save_backend == 'pil':
      if len(self.img.shape) == 3:
        PIL.Image.fromarray(self.img[:, :, ::-1]).save(
          path, exif={TAGS_INV[key]: val for key, val in self.metadata.items()
                      if key in TAGS_INV})
      else:
        PIL.Image.fromarray(self.img).save(
          path, exif={TAGS_INV[key]: val for key, val in self.metadata.items()
                      if key in TAGS_INV})

    elif self._save_backend == 'cv2':
      cv2.imwrite(path, self.img)

    elif self._save_backend == 'npy':
      np.save(path, self.img)

    # Sending the results to the downstream Blocks
    if self._send_msg:
      self.send({'t(s)': self.metadata['t(s)'],
                 'img_index': self.metadata['ImageUniqueID'],
                 'meta': self.metadata})

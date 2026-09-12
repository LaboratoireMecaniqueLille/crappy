# coding: utf-8

from pathlib import Path
from typing import Literal, Any
import logging
import numpy as np
from csv import DictWriter

from .block import VisionBlock
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


class ImageRecorder(VisionBlock):
  """This :class:`~crappy.blocks.vision.VisionBlock` can record images acquired
  by a :class:`~crappy.blocks.Camera` Block to the desired location and in the
  desired format.

  The images are received through a :class:`~crappy.links.ImageLink`. Whenever
  an image is saved, its information is sent through the downstream
  :class:`~crappy.links.Link` if any, containing the timestamp, the
  image index, and the metadata. They are respectively carried by the `'t(s)'`,
  `'img_index'` and `'meta'` labels. This is useful for performing an action
  conditionally at each new saved image.

  Various backends can be used for recording the images, some may be faster or
  slower depending on the machine. It is possible to only save one out of a
  given number of images, if not all frames are needed.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               img_extension: str = "tiff",
               save_folder: str | Path | None = None,
               save_period: int = 1,
               save_backend: Literal['sitk', 'pil',
                                     'cv2', 'npy'] | None = None,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 100) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      img_extension: The file extension for the recorded images, as a
        :obj:`str` and without the dot. Common file extensions include `tiff`,
        `png`, `jpg`, etc.
      save_folder: :obj:`pathlib.Path` to the folder where to save the images.
        Can be an absolute or a relative path. The folder does not need to
        already exist, in which case it is created.
      save_period: Only one out of that number (as an :obj:`int`) images at
        most will be saved. Allows to have a known periodicity in case the
        framerate is too high to record all the images. Or simply to reduce the
        number of saved images if saving them all is not needed.
      save_backend: The backend to use for saving the images. Should be one of:
        ::

          'sitk', 'pil', 'cv2', 'npy'

        They correspond to the modules :mod:`SimpleITK`, :mod:`PIL` (Pillow
        Fork), :mod:`cv2` (OpenCV), and :mod:`numpy`. Depending on the machine,
        some may be faster or slower. The ``img_extension`` is ignored for the
        backend ``'npy'``, that saves the images as raw numpy arrays.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
    """

    super().__init__(img_shape=None,
                     img_dtype=None,
                     display_freq=display_freq,
                     debug=debug,
                     freq=freq)

    # Validate arguments before setting them
    if not img_extension and (save_backend is None or save_backend != 'npy'):
      raise ValueError("img_extension must be a non-empty string")
    if (save_folder is not None and
        ((not isinstance(save_folder, str) or not save_folder)
         and not isinstance(save_folder, Path))):
      raise ValueError("When provided, save_folder must be a non-empty string "
                       "or a Path")
    if not isinstance(save_period, int) or save_period < 1:
      raise ValueError("save_period must be a strictly positive integer")
    if save_backend is not None and save_backend not in ('sitk', 'pil',
                                                         'cv2', 'npy'):
      raise ValueError("When provided, save_backend must be one of 'sitk', "
                       "'pil', 'cv2', 'npy'")

    # Trying the different possible backends and checking if the given one
    # is correct
    if save_backend is None:
      if not isinstance(Sitk, OptionalModule):
        self._save_backend: str = 'sitk'
      elif not isinstance(PIL, OptionalModule):
        self._save_backend: str = 'pil'
      elif not isinstance(cv2, OptionalModule):
        self._save_backend: str = 'cv2'
      else:
        self._save_backend: str = 'npy'
    elif save_backend in ('sitk', 'pil', 'cv2', 'npy'):
      if save_backend == 'sitk' and isinstance(Sitk, OptionalModule):
        raise ModuleNotFoundError("Backend 'sitk' requested but could not "
                                  "be imported")
      elif save_backend == 'pil' and isinstance(PIL, OptionalModule):
        raise ModuleNotFoundError("Backend 'pil' requested but could not "
                                  "be imported")
      elif save_backend == 'cv2' and isinstance(cv2, OptionalModule):
        raise ModuleNotFoundError("Backend 'cv2' requested but could not "
                                  "be imported")

      self._save_backend: str = save_backend
    else:
      raise ValueError("The save_backend argument should be either 'sitk', "
                       "'pil', 'cv2' or 'npy'!")

    # The image extension cannot be chosen when the Numpy backend is used
    self._img_extension: str = (img_extension if self._save_backend != 'npy'
                                else '')

    # The default save folder is in the current working directory
    if save_folder is None:
      self._save_folder: Path = Path.cwd() / 'Crappy_images'
    else:
      self._save_folder: Path = Path(save_folder)

    self._save_period: int = save_period

    # Other attributes that will be useful later
    self._csv_created = False
    self._csv_path = None
    self._metadata_name = 'metadata.csv'
    self._last_processed_idx: int = -1

  def prepare(self) -> None:
    """Creates the folder for saving the images.

    If a folder is already present at the indicated path and contains images,
    saving to a new folder with the same name but ending with a suffix.
    """

    # Ensuring Link consistency
    if self.inputs:
      raise IOError("This Block does not accept input Links")
    if self.img_outputs:
      raise IOError("This VisionBlock does not support output ImageLink")
    if not self.img_inputs:
      raise IOError("This VisionBlock is useless without an input ImageLink")
    if len(self.img_inputs) != 1:
      raise IOError("This VisionBlock requires exactly one input ImageLink")

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

    super().prepare()

  def loop(self) -> None:
    """This method grabs the latest frame, writes its metadata to a `.csv` file
    and saves the image at the chosen location using the chosen backend.

    On the first frame, the metadata file is created and its header is
    populated using the metadata of the frame.

    Frames may be skipped depending on th echosen save period.
    """

    # Nothing to do if no new image was received
    if not (upd_links := self.receive_imgs()):
      self.log(logging.DEBUG, "No new image received during this loop")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Get the ImageLink name
    upd_link, = upd_links

    # If not enough images have arrived for the given period, not saving
    img_idx = self.last_received[upd_link].id
    if (self._last_processed_idx >= 0 and
        self.last_received[upd_link].id - self._last_processed_idx <
        self._save_period):
      self.log(logging.DEBUG, "Not processing because haven't reached the "
                              "save period yet")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Storing the latest index
    self._last_processed_idx = img_idx

    # Defined for convenience
    metadata = self.last_received[upd_link].metadata
    if metadata is None:
      raise RuntimeError("At that point, the image metadata should not be "
                         "empty")
    img = self.last_received[upd_link].img

    # Creating the .csv containing the metadata on the first received frame
    if not self._csv_created:
      self._csv_path = (self._save_folder / self._metadata_name)

      self.log(logging.INFO, f"Creating file for saving the metadata: "
                             f"{self._csv_path}")

      # Also writing the header of the .csv file when creating it
      with open(self._csv_path, 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=metadata.keys())
        writer.writeheader()

      self._csv_created = True

    # Saving the received metadata to the .csv file
    self.log(logging.DEBUG, f"Saving metadata: {metadata}")
    with open(self._csv_path, 'a') as csvfile:
      writer = DictWriter(csvfile, fieldnames=metadata.keys())
      writer.writerow({**metadata, 't(s)': metadata['t(s)']})

    # Only include the extension for the image file if applicable
    if self._img_extension:
      path = str(self._save_folder / f"{metadata['ImageUniqueID']:06d}_"
                                     f"{metadata['t(s)']:.3f}."
                                     f"{self._img_extension}")
    else:
      path = str(self._save_folder / f"{metadata['ImageUniqueID']:06d}_"
                                     f"{metadata['t(s)']:.3f}")

    # Saving the image at the destination path using the chosen backend
    self.log(logging.DEBUG, "Saving image")
    if self._save_backend == 'sitk':
      if len(img.shape) == 3:
        Sitk.WriteImage(Sitk.GetImageFromArray(img[:, :, ::-1],
                                               isVector=True), path)
      else:
        Sitk.WriteImage(Sitk.GetImageFromArray(img), path)

    elif self._save_backend == 'pil':
      if len(img.shape) == 3:
        PIL.Image.fromarray(img[:, :, ::-1]).save(
            path, exif=self._pil_exif(metadata))
      else:
        PIL.Image.fromarray(img).save(path, exif=self._pil_exif(metadata))

    elif self._save_backend == 'cv2':
      cv2.imwrite(path, img)

    elif self._save_backend == 'npy':
      np.save(path, img)

      # Sending information on the image through regular Links
      if self.last_received[upd_link].metadata is None:
        raise RuntimeError("At that point, the image metadata should not be "
                           "empty")
      if ('t(s)' not in self.last_received[upd_link].metadata
          and 'ImageUniqueID' not in self.last_received[upd_link].metadata):
        raise RuntimeError(
          "At that point, 't(s)' and 'ImageUniqueID' should be "
          "in the metadata dictionary")
      self.send({'t(s)': metadata['t(s)'],
                 'img_index': metadata['ImageUniqueID'],
                 'meta': metadata})

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def _pil_exif(self, metadata: dict[str, Any]):
    """Parses the metadata of the current image and converts it to a
    PIL.Image.Exif object."""

    exif = PIL.Image.Exif()

    for key, value in metadata.items():
      if key not in TAGS_INV:
        continue

      # Convert Numpy scalars to Python object
      if isinstance(value, np.generic):
        value = value.item()

      # These EXIF fields are ASCII in practice
      if key in ('ImageUniqueID', 'SubsecTimeOriginal'):
        value = str(value)

      try:
        exif[TAGS_INV[key]] = value
      except (TypeError, ValueError):
        self.log(logging.DEBUG, f"Could not encode metadata field "
                                f"{key} as EXIF")

    return exif

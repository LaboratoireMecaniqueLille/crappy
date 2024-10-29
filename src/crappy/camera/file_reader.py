# coding: utf-8

from time import time, sleep
from typing import Union, Optional
import numpy as np
from pathlib import Path
from re import fullmatch
import logging

from .meta_camera import Camera
from .._global import OptionalModule, ReaderStop

try:
  import SimpleITK as Sitk
except (ModuleNotFoundError, ImportError):
  Sitk = OptionalModule("SimpleITK")
try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class FileReader(Camera):
  """This Camera class reads existing images from a given folder, in the same
  order in which they were acquired.

  The name of the images to read must follow the following pattern :
  ``<frame_nr>_<frame_seconds>.<frame_subseconds>.<file_extension>``. This
  pattern is the same as used for recording images with the
  :class:`~crappy.blocks.Camera` of Crappy, so images recorded via Crappy are
  readily readable and don't need to be re-named.

  This class tries to read the images at the same framerate as they were
  recorded, although the control of the framerate is not so precise. It might
  be that the images cannot be read fast enough to match the original
  framerate, in which case the images are read as fast as possible and the
  delay keeps growing.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10 renamed from *Streamer* to *File_reader*
  .. versionchanged:: 2.0.0 renamed from *File_reader* to *FileReader*
  """

  def __init__(self) -> None:
    """Initializes the parent class and sets a few attributes."""

    super().__init__()

    # These attributes will come in use later on
    self._images = None
    self._stop_at_end = True
    self._backend = None
    self._t0 = None
    self._stopped = False

  def open(self,
           reader_folder: Union[Path, str],
           reader_backend: Optional[str] = None,
           stop_at_end: bool = True) -> None:
    """Sets the reader backend and retrieves the images to read, sorted by
    their timestamp.

    Args:
      reader_folder: The path to the folder containing the images to read.
      
        .. versionchanged:: 1.5.10 renamed from *path* to *reader_folder*
      reader_backend: The backend to use for reding the images. Should be one
        of :
        ::

          'sitk' or 'cv2'

        If not given, SimpleITK is preferred over OpenCV if available.

        .. versionadded:: 1.5.10
      stop_at_end: If :obj:`True` (the default), stops the Crappy script once
        the available images are all exhausted. Otherwise, simply remains idle
        while waiting for the test to finish.

        .. versionadded:: 1.5.10

    .. versionremoved::
       1.5.10 *pattern*, *start_delay* and *modifier* arguments
    """

    # Selecting an  available backend between first sitk and then cv2
    if reader_backend is None:
      if not isinstance(Sitk, OptionalModule):
        self._backend = 'sitk'
      elif not isinstance(cv2, OptionalModule):
        self._backend = 'cv2'
      else:
        raise ModuleNotFoundError("Neither SimpleITK nor opencv-python could "
                                  "be imported, no backend found for reading "
                                  "the images")
    # Setting the backend requested by the user
    elif reader_backend in ('sitk', 'cv2'):
      self._backend = reader_backend
    else:
      raise ValueError("The backend argument should be either 'sitk' or "
                       "'cv2' !")

    self._stop_at_end = stop_at_end

    # Making sure that the given folder is valid
    folder = Path(reader_folder)
    if not folder.exists() or not folder.is_dir():
      raise FileNotFoundError(f"The {folder} folder does not exist or is not "
                              f"a folder !")

    # Retrieving all the images in the given folder that match the name pattern
    images = (path for path in folder.glob('*') if
              fullmatch(r'\d+_\d+\.\d+\..+\Z', path.name) is not None)
    # Sorting the images by timestamp
    images = sorted(images,
                    key=lambda p: float(fullmatch(r'\d+_(\d+\.\d+)',
                                                  p.stem).group(1)))

    # In case no matching image was found
    if not images:
      raise FileNotFoundError(f"Could not find matching images in the {folder}"
                              f" folder !\nPlease specify a valid folder or "
                              f"put the image names in the right format.")

    self.log(logging.INFO, f"Detected {len(images)} images in the {folder} "
                           f"folder")

    # The images are stored as an iterator
    self._images = iter(images)

  def get_image(self) -> Optional[tuple[float, np.ndarray]]:
    """Reads the next image in the image folder, and returns it at the right
    time so that the achieved framerate matches the original framerate.

    If the original framerate cannot be achieved, just reads the image as fast
    as possible.

    By default, stops the test when there's no image left to read. If specified
    otherwise, just remains idle until the test ends.
    """

    # Setting the approximate start time (potentially not well synced with the
    # actual t0 of Crappy's blocks)
    if self._t0 is None:
      self._t0 = time()

    if self._stopped:
      sleep(0.1)
      return

    try:
      # Getting the next image to read and its timestamp
      img_path = next(self._images)
      timestamp = float(fullmatch(r'\d+_(\d+\.\d+)', img_path.stem).group(1))

      self.log(logging.DEBUG, f"Reading image {img_path} with timestamp "
                              f"{timestamp}")

      # Reading the image data with the chosen backend
      if self._backend == 'sitk':
        img = Sitk.GetArrayFromImage(Sitk.ReadImage(img_path))
      else:
        img = cv2.imread(str(img_path), 0)

      # Delaying the return of the image if we're ahead of time
      t = time()
      if t - self._t0 < timestamp:
        sleep(timestamp - (t - self._t0))

      return t, img

    # Raised when there's no more image to read
    except StopIteration:
      # Default behavior, stop the test
      if self._stop_at_end:
        raise ReaderStop
      else:
        # Otherwise, nothing more gets done but the test goes on
        self._stopped = True
        self.log(logging.WARNING, "Exhausted all the images to read for the "
                                  "FileReader camera, staying idle until the "
                                  "script ends")

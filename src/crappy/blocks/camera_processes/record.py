# coding: utf-8

from csv import DictWriter
import numpy as np
from pathlib import Path
import logging
import logging.handlers

from ... import _validation as check
from .camera_process import CameraProcess
from ..._global import OptionalModule

Sitk = OptionalModule('SimpleITK', lazy_import=True)
Image = OptionalModule('PIL.Image', install_name='Pillow', lazy_import=True)
ExifTags = OptionalModule('PIL.ExifTags', install_name='Pillow',
                          lazy_import=True)
cv2 = OptionalModule('cv2', install_name='opencv-python', lazy_import=True)


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
               save_folder: str | Path | None = None,
               save_period: int = 1,
               save_backend: str | None = None,
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

        They correspond to the modules ``SimpleITK``, :mod:`PIL` (Pillow
        Fork), ``cv2`` (OpenCV), and :mod:`numpy`. Depending on the machine,
        some may be faster or slower. The ``img_extension`` is ignored for the
        backend ``'npy'``, that saves the images as raw numpy arrays.
        Only the requested backend is checked, otherwise SimpleITK, Pillow,
        OpenCV, and NumPy are tried in that order. The selected backend is
        initialized in the recording process before any frames are saved.
      send_msg: In case no processing is performed, and if output Links are
        present, this argument is set to :obj:`True`. In that case, a message
        containing the timestamp, the index, and the metadata of the image is
        sent to downstream Blocks each time an image is saved.

        .. versionadded:: 2.0.5
    """

    super().__init__()

    check.boolean(send_msg, 'send_msg')

    # Probe only the requested backend, or stop at the first available one
    backends = {'sitk': Sitk.is_available,
                'pil': lambda: (Image.is_available() and
                                ExifTags.is_available()),
                'cv2': cv2.is_available,
                'npy': lambda: True}
    if save_backend is None:
      for name, is_available in backends.items():
        if is_available():
          self._save_backend = name
          break

    else:
      if not backends[save_backend]():
        raise ModuleNotFoundError(f"Backend {save_backend!r} requested but "
                                  "could not be imported")
      self._save_backend = save_backend

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
    self._pil_tags_inv: dict[str, int] = dict()

  def init(self) -> None:
    """Resolves the selected backend and creates the recording folder.

    If a folder is already present at the indicated path and contains images,
    saving to a new folder with the same name but ending with a suffix.
    """

    # Fresh workers must finish importing before handling their first frame
    if self._save_backend == 'sitk':
      Sitk.resolve()
    elif self._save_backend == 'pil':
      Image.resolve().init()
      self._pil_tags_inv = {val: key for key, val
                            in ExifTags.resolve().TAGS.items()}
    elif self._save_backend == 'cv2':
      cv2.resolve()

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

  def _has_new_image(self) -> bool:
    """Waits for a frame to save, not for the skipped frames.

    .. versionadded:: 2.2.0
    """

    assert self._data_dict is not None
    image_id = self._data_dict.get('ImageUniqueID')
    previous_id = self.metadata['ImageUniqueID']
    return (image_id is not None and
            (previous_id is None or
             image_id - previous_id >= self._save_period))

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
        Image.fromarray(self.img[:, :, ::-1]).save(path,
                                                   exif=self._pil_exif())
      else:
        Image.fromarray(self.img).save(path, exif=self._pil_exif())

    elif self._save_backend == 'cv2':
      cv2.imwrite(path, self.img)

    elif self._save_backend == 'npy':
      np.save(path, self.img)

    # Sending the results to the downstream Blocks
    if self._send_msg:
      self.send({'t(s)': self.metadata['t(s)'],
                 'img_index': self.metadata['ImageUniqueID'],
                 'meta': self.metadata})

  def _pil_exif(self):
    """Parses the metadata of the current image and converts it to a
    PIL.Image.Exif object."""

    exif = Image.Exif()

    for key, value in self.metadata.items():
      if key not in self._pil_tags_inv:
        continue

      # Convert Numpy scalars to Python object
      if isinstance(value, np.generic):
        value = value.item()

      # These EXIF fields are ASCII in practice
      if key in ('ImageUniqueID', 'SubsecTimeOriginal'):
        value = str(value)

      try:
        exif[self._pil_tags_inv[key]] = value
      except (TypeError, ValueError):
        self.log(logging.DEBUG, f"Could not encode metadata field "
                                f"{key} as EXIF")

    return exif

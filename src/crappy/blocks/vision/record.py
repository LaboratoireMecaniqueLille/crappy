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
  """Records an image stream and its metadata to disk.

  This Block receives images from exactly one upstream
  :class:`~crappy.blocks.vision.block.VisionBlock` through an input
  :class:`~crappy.links.img_link.ImageLink`. It accepts no regular input Link
  and has no output ImageLink. The latest eligible frame is saved on each
  handling cycle, so a slow recorder may skip intermediate frames that were
  overwritten in the source's shared buffer. ``save_period`` can deliberately
  reduce the recording rate further by retaining at most one out of a given
  number of source images.

  SimpleITK, Pillow, OpenCV, and raw NumPy output are supported. When no
  backend is requested, the first available backend is selected in that order,
  with NumPy always available as a fallback. NumPy output uses ``.npy`` files
  and ignores ``img_extension``. Pillow additionally embeds compatible metadata
  fields as EXIF data.

  Images are stored under a filename containing their zero-padded
  ``'ImageUniqueID'`` and ``'t(s)'`` timestamp. A ``metadata.csv`` file records
  the complete metadata of each saved frame. If the requested folder already
  contains such a file, a new sibling folder with a numeric suffix is chosen so
  existing recordings are not overwritten.

  Whenever a saved-frame notification is emitted through regular output
  :class:`~crappy.links.link.Link` objects, it contains the image timestamp,
  unique ID, and complete metadata under ``'t(s)'``, ``'img_index'``, and
  ``'meta'``. This allows downstream actions to depend on images selected for
  recording.

  Unlike :class:`~crappy.blocks.camera_processes.ImageSaver`, which is managed
  internally by the older :class:`~crappy.blocks.Camera`, this class is an
  independent Block that can record images from any compatible VisionBlock.

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
    """Sets the destination, image format, and recording rate.

    Args:
      img_extension: File extension used for encoded images, without the
        leading dot. Common values include ``'tiff'``, ``'png'``, and
        ``'jpg'``. It is ignored by the ``'npy'`` backend.
      save_folder: Absolute or relative directory in which recordings are
        stored. Missing directories are created. If omitted, images are saved
        in ``Crappy_images`` under the current working directory. A numeric
        suffix is added when the target already contains Crappy metadata.
      save_period: Saves at most one image for every this many transport-level
        source image identifiers. It must be a strictly positive integer. A
        value of one considers every newly received frame.
      save_backend: Image-writing backend, chosen from:
        ::

          'sitk', 'pil', 'cv2', 'npy'

        These correspond to :mod:`SimpleITK`, :mod:`PIL`, :mod:`cv2`, and
        :mod:`numpy`. If omitted, they are tried in that order and the first
        installed backend is selected. Requesting an unavailable optional
        backend raises :exc:`ModuleNotFoundError`.
      display_freq: If :obj:`True`, periodically reports the achieved image
        recording frequency.
      debug: If :obj:`True`, displays all log messages, including
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: Target frequency for checking and recording new images. If
        :obj:`None`, loops as fast as possible. Storage performance and the
        source frequency may limit the actual recording rate.
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
    """Validates the topology and prepares the recording directory.

    The Block requires exactly one input ImageLink, accepts no regular input
    Link, and supports no output ImageLink. If the target directory already
    contains ``metadata.csv``, a free sibling name ending in ``_00001``,
    ``_00002``, and so on is selected. The directory is then created when
    necessary before the upstream shared image buffer is attached.

    Raises:
      IOError: If the Link or ImageLink topology is unsupported.
      OSError: If the recording directory cannot be inspected or created.
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
    """Saves the newest frame selected by the configured period.

    If no new image is available, or fewer than ``save_period`` source image
    identifiers separate it from the last saved frame, this method returns
    without writing. For the first saved frame, ``metadata.csv`` is created and
    its header is populated from that frame's metadata keys. Every selected
    frame appends one metadata row and is written using a filename of the form
    ``<ImageUniqueID>_<timestamp>.<extension>``. The extension is omitted from
    the requested path for NumPy output, allowing :func:`numpy.save` to append
    ``.npy``.

    Saved-frame information is also sent through regular output Links under
    ``'t(s)'``, ``'img_index'``, and ``'meta'``.

    Raises:
      RuntimeError: If copied image metadata is unavailable.
      KeyError: If metadata lacks ``'t(s)'`` or ``'ImageUniqueID'``.
      OSError: If metadata or image files cannot be written.
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
    """Converts compatible image metadata to Pillow EXIF fields.

    Unknown fields and values that Pillow cannot encode are skipped.
    :mod:`numpy` scalar values are converted to their native Python
    equivalents first.

    Args:
      metadata: Metadata associated with the image being saved.

    Returns:
      A :class:`PIL.Image.Exif` object containing encodable known fields.
    """

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

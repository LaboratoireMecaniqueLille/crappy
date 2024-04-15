# coding: utf-8

import numpy as np
from io import BytesIO
import time
from typing import Optional, Tuple, Dict, Any

from .meta_camera import Camera
from .._global import OptionalModule

try:
  from PIL import Image, ExifTags
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("Pillow")
  ExifTags = OptionalModule("Pillow")

try:
  import gphoto2 as gp
except (ModuleNotFoundError, ImportError):
  gp = OptionalModule("gphoto2")


class CameraGPhoto2(Camera):
  """Class for reading images from agphoto2 compatible Camera.

  The CameraGphoto2 block is meant for reading images from a
  Gphoto2 Camera. It uses the :mod:`gphoto2` library for capturing images,
  and :mod:`PIL` for converting BGR images to black and white.

  Read images from the all the gphoto2 compatible cameras  indifferently.

  2 modes are currently implemented :
      'continuous' : take picture as fast as possible
      'hardware_trigger' : take picture when button is clicked

  Warning:
    Not tested in Windows, but there is no use of Linux API,
     only python libraries.
  .. versionadded:: ?
  """

  def __init__(self) -> None:
    """Instantiates the available settings."""

    super().__init__()

    self._camera: Optional[gp.Camera] = None
    self._context: gp.GPContext = gp.Context()
    self._model: Optional[str] = None
    self._port: Optional[str] = None
    self._num_image = 0

    self.add_choice_setting(name="channels",
                            choices=('1', '3'),
                            getter=None,
                            setter=None,
                            default='1')
    self.add_choice_setting(name="mode",
                            choices=('continuous', 'hardware_trigger'),
                            getter=None,
                            setter=None,
                            default='continuous')

  def open(self,
           model: Optional[str] = None,
           port: Optional[str] = None,
           **kwargs: any) -> None:
    """Open the camera `model` and `could be specified`"""

    self._model = model
    self._port = port
    self.set_all(**kwargs)

    cameras = gp.Camera.autodetect(self._context)
    _port_info_list = gp.PortInfoList()
    _port_info_list.load()

    camera_found = False
    for name, port in cameras:
      if ((self._model is None or name == self._model) and
          (self._port is None or port == self._port)):
        idx = _port_info_list.lookup_path(port)
        if idx >= 0:
          self._camera = gp.Camera()
          self._camera.set_port_info(_port_info_list[idx])
          self._camera.init(self._context)
          camera_found = True
          break

    if not camera_found:
      if self._model is not None and self._port is not None:
        raise IOError(
          f"Camera '{self._model}' on port '{self._port}' not found."
        )
      elif self._model is not None and self._port is None:
        raise IOError(f"Camera '{self._model}' not found.")
      elif self._model is None and self._port is None:
        raise IOError(f"No camera found found.")

  def get_image(self) -> Tuple[Dict[str, Any], np.ndarray]:
    """Simply acquire an image using gphoto2 library.
    The captured image is in GBR format, and converted into black and white if
    needed.
    Returns:
    The timeframe and the image.
    """

    if self.mode == 'hardware_trigger':
      # Wait for a hardware trigger event
      print("Waiting for hardware trigger...")
      event_type, event_data = self._camera.wait_for_event(200, self._context)
      if event_type == gp.GP_EVENT_FILE_ADDED:
        camera_file_path = event_data
        camera_file = gp.CameraFile()
        self._camera.file_get(camera_file_path.folder,
                              camera_file_path.name,
                              gp.GP_FILE_TYPE_NORMAL,
                              camera_file,
                              self._context)
      else:
        return
    else:
      file_path = self._camera.capture(gp.GP_CAPTURE_IMAGE, self._context)
      camera_file = self._camera.file_get(
          file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)

    file_data = camera_file.get_data_and_size()
    image_stream = BytesIO(file_data)
    img = Image.open(image_stream)
    # Extract EXIF data
    t = time.time()
    # Extract and interpret EXIF data
    metadata = {}
    if hasattr(img, '_getexif'):
      exif_info = img._getexif()
      if exif_info is not None:
        for tag, value in exif_info.items():
          decoded = ExifTags.TAGS.get(tag, tag)
          if decoded in ["Model", "DateTime", "ExposureTime",
                         "ShutterSpeedValue", "FNumber", "ApertureValue",
                         "FocalLength", "ISOSpeedRatings"]:
            metadata[decoded] = value
    metadata = {'ImageUniqueID': self._num_image, **metadata}
    metadata = {'t(s)': t, **metadata}
    self._num_image += 1
    if self.channels == '1':
      img = img.convert('L')
      metadata['channels'] = 'gray'
      return metadata, np.array(img)
    else:
      metadata['channels'] = 'color'
      img = np.array(img)
      return metadata, img[:, :, ::-1]

  def close(self) -> None:
    """Close the camera in gphoto2 library"""

    if self._camera is not None:
      self._camera.exit(self._context)

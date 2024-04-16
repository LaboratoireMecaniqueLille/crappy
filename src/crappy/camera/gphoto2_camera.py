# coding: utf-8

import numpy as np
from io import BytesIO
from time import time
from typing import Optional, Tuple, Dict, Any
import logging

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

    # Attributes used for image acquisition
    self._camera: Optional[gp.Camera] = None
    self._context: gp.GPContext = gp.Context()
    self._model: Optional[str] = None
    self._port: Optional[str] = None
    self._num_image = 0

    # Basic settings always implemented
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
    """Open the camera `model` and `could be specified`"""
           channels: str = '1',
           mode: str = 'continuous') -> None:
    The available connected cameras are first scanned, and if one matches the
    model and port requirements it is opened.

    # Detecting the connected and compatible cameras
    cameras = gp.Camera.autodetect(self._context)

    # Listing all the available ports
    port_info_list = gp.PortInfoList()
    port_info_list.load()

    # Checking if a camera matches the port and/or model specifications and
    # instantiating it if so
    # If nothing specified, instantiating the first camera found
    for name, detected_port in cameras:
      self.log(logging.DEBUG, f"Detected camera {name} on port "
                              f"{detected_port}")
      if ((model is None or name == model) and
          (port is None or detected_port == port)):
        idx = port_info_list.lookup_path(detected_port)

        if idx >= 0:
          self.log(logging.INFO, f"Instantiating camera {name} on port "
                                 f"{detected_port}")
          self._camera = gp.Camera()
          self._camera.set_port_info(port_info_list[idx])
          self._camera.init(self._context)
          break

    # Raising an exception in case no compatible camera was found
    if self._camera is None:
      if model is not None and port is not None:
        raise IOError(f"Could not find camera {model} on port {port} !")
      elif model is not None and port is None:
        raise IOError(f"Could not find camera {model} !")
      elif model is None and port is not None:
        raise IOError(f"Could not find a camera on port {port} !")
      else:
        raise IOError(f"No compatible camera found !")

    # Not currently used, but might be needed in the future
    self.set_all(channels=channels, mode=mode)

  def get_image(self) -> Tuple[Dict[str, Any], np.ndarray]:
    """Simply acquire an image using gphoto2 library.
    The captured image is in GBR format, and converted into black and white if
    needed.
    Returns:
    The timeframe and the image.
    """

    if self.mode == 'hardware_trigger':
      # Wait for a hardware trigger event
      event_type, event_data = self._camera.wait_for_event(200, self._context)
      # Get the acquired image if a new one was captured
      if event_type == gp.GP_EVENT_FILE_ADDED:
        camera_file = gp.CameraFile()
        self._camera.file_get(event_data.folder,
                              event_data.name,
                              gp.GP_FILE_TYPE_NORMAL,
                              camera_file,
                              self._context)
        self.log(logging.DEBUG, f"One new image grabbed in this loop, file "
                                f"in {event_data.folder}/{event_data.name}")
      # Otherwise, not returning anything
      else:
        self.log(logging.DEBUG, f"No new image captured in this loop, got "
                                f"event type: {event_type}")
        return

    # In continuous mode, getting the image in all cases
    elif self.mode == 'continuous':
      file_path = self._camera.capture(gp.GP_CAPTURE_IMAGE, self._context)
      camera_file = self._camera.file_get(
          file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)

    else:
      self.log(logging.WARNING, f"The acquisition mode {self.mode} is not "
                                f"supported, aborting")
      return

    # Getting the image from the gPhoto2 buffer
    img = Image.open(BytesIO(camera_file.get_data_and_size()))

    # Building the metadata dictionary
    metadata = dict()
    exif_info = img.getexif()
    for tag, value in exif_info.items():
      decoded = ExifTags.TAGS.get(tag, tag)
      if decoded in ["Model", "DateTime", "ExposureTime",
                     "ShutterSpeedValue", "FNumber", "ApertureValue",
                     "FocalLength", "ISOSpeedRatings"]:
        metadata[decoded] = value
    metadata = {'ImageUniqueID': self._num_image, 't(s)': time(), **metadata}
    self._num_image += 1

    # Casting the image to grey level if needed
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
      self.log(logging.INFO, "Closing the camera object.")
      self._camera.exit(self._context)

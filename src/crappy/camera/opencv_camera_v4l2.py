# coding: utf-8

from time import time, sleep
from typing import Optional
from numpy import ndarray
from subprocess import run
from re import findall, search
import logging

from .meta_camera import Camera
from ._v4l2_base import V4L2Helper
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class CameraOpencv(Camera, V4L2Helper):
  """A class for reading images from any camera able to interface with OpenCV.

  The number of the video device to read images from can be specified.

  This camera class is less performant than the
  :class:`~crappy.camera.CameraGstreamer` one that relies on GStreamer, but the
  installation of OpenCV is way easier than the one of GStreamer, especially on
  Windows !

  Warning:
    There are two classes for CameraOpencv, one for Linux based on
    `v4l-utils`, and another one for Linux (without `v4l-utils`) and other OS.
    Depending on the installation of `v4l-utils` and the OS, the correct class
    will be automatically imported. The version using `v4l-utils` allows tuning
    more parameters than the basic version.

  .. versionadded:: 1.5.9
  .. versionchanged:: 2.0.0 renamed from *Camera_opencv* to *CameraOpencv*
  """

  def __init__(self) -> None:
    """Sets variables and adds the channels setting."""

    Camera.__init__(self)
    V4L2Helper.__init__(self)

    self._cap = None
    self._device_num: Optional[int] = None
    self._formats: list[str] = list()

  def open(self, device_num: int = 0, **kwargs) -> None:
    """Opens the video stream and sets any user-specified settings.

    Args:
      device_num: The index of the device to open, as an :obj:`int`.
      **kwargs: Any additional setting to set before opening the configuration
        window.
    """

    if not isinstance(device_num, int) or device_num < 0:
      raise ValueError("device_num should be an integer !")

    # Opening the videocapture device
    self.log(logging.INFO, "Opening the image stream from the camera")
    self._cap = cv2.VideoCapture(device_num)
    self._device_num = device_num

    # Getting the available formats for the selected device and filtering the
    # supported ones
    self._get_available_formats(device_num)
    supported = ('MJPG', 'YUYV')
    unavailable = set([_format.split()[0] for _format in self._formats
                       if _format.split()[0] not in supported])
    if unavailable:
      self.log(logging.WARNING, f"The formats {', '.join(unavailable)} "
                                f"are available but not implemented in Crappy")
    self._formats = [_format for _format in self._formats
                     if _format.split()[0] in supported]

    # Instantiating the format setting if there are formats left
    if self._formats:
      self.add_choice_setting(name='format',
                              choices=tuple(self._formats),
                              getter=self._get_format_size,
                              setter=self._set_format)

    # Getting the available parameters for the camera
    self._get_param(device_num)

    # Creating the different settings
    for param in self._parameters:
      if param.type == 'int':
        self.add_scale_setting(
          name=param.name,
          lowest=int(param.min),
          highest=int(param.max),
          getter=self._add_scale_getter(param.name, self._device_num),
          setter=self._add_setter(param.name, self._device_num),
          default=param.default,
          step=int(param.step))

      elif param.type == 'bool':
        self.add_bool_setting(
          name=param.name,
          getter=self._add_bool_getter(param.name, self._device_num),
          setter=self._add_setter(param.name, self._device_num),
          default=bool(int(param.default)))

      elif param.type == 'menu':
        if param.options:
          self.add_choice_setting(
            name=param.name,
            choices=param.options,
            getter=self._add_menu_getter(param.name, self._device_num),
            setter=self._add_setter(param.name, self._device_num),
            default=param.default)

      else:
        self.log(logging.ERROR, f'The type {param.type} is not yet'
                                f' implemented. Only int, bool and menu '
                                f'type are implemented. ')
        raise NotImplementedError

    self.add_choice_setting(name="channels", choices=('1', '3'), default='1')

    # Adding the software ROI selection settings
    if 'format' in self.settings:
      width, height = search(r'(\d+)x(\d+)', self._get_format_size()).groups()
      self.add_software_roi(int(width), int(height))

    # Setting the kwargs if any
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, ndarray]:
    """Grabs a frame from the videocapture object and returns it along with a
    timestamp."""

    # Grabbing the frame and the timestamp
    t = time()
    ret, frame = self._cap.read()

    # Checking the integrity of the frame
    if not ret:
      raise IOError("Error reading the camera")

    # Returning the image in the right format, and its timestamp
    if self.channels == '1':
      return t, self.apply_soft_roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
      return t, self.apply_soft_roi(frame)

  def close(self) -> None:
    """Releases the videocapture object."""

    if self._cap is not None:
      self.log(logging.INFO, "Closing the image stream from the camera")
      self._cap.release()

  def _get_width(self) -> int:
    """Returns the current image width."""

    return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_height(self) -> int:
    """Returns the current image height."""

    return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_width(self, width: int) -> None:
    """Tries to set the image width."""

    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = self._get_width(), self._get_height()
      self.reload_software_roi(width, height)

  def _set_height(self, height: int) -> None:
    """Tries to set the image height."""

    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = self._get_width(), self._get_height()
      self.reload_software_roi(width, height)

  def _set_format(self, img_format: str) -> None:
    """Sets the format of the image according to the user's choice."""

    # The format is made of a name, a size and a framerate
    format_name, img_size, fps = findall(r"(\w+)\s(\w+)\s\((\d+.\d+) fps\)",
                                         img_format)[0]

    # Setting the format
    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format_name))

    # Getting the width and height from the second half of the string
    width, height = map(int, img_size.split('x'))

    # Setting the size
    self._set_width(width)
    self._set_height(height)

    # Setting the acquisition frequency
    self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = search(r'(\d+)x(\d+)', self._get_format_size()).groups()
      self.reload_software_roi(int(width), int(height))

  def _get_format_size(self) -> str:
    """Parses the ``v4l2-ctl -all`` command to get the current image format as
    a :obj:`str`."""

    # Sending the v4l2-ctl command
    command = ['v4l2-ctl', '-d', str(self._device_num), '--all']
    self.log(logging.DEBUG, f"Getting the current image formats with "
                            f"command {' '.join(command)}")
    ret = run(command, capture_output=True, text=True).stdout
    self.log(logging.DEBUG, f"Got the following image formats: {ret}")

    # Parsing the answer
    format_ = width = height = fps = ''
    if search(r"Pixel Format\s*:\s*'(\w+)'", ret) is not None:
      format_, *_ = search(r"Pixel Format\s*:\s*'(\w+)'", ret).groups()
    if search(r"Width/Height\s*:\s*(\d+)/(\d+)", ret) is not None:
      width, height = search(r"Width/Height\s*:\s*(\d+)/(\d+)", ret).groups()
    if search(r"Frames per second\s*:\s*(\d+.\d+)", ret) is not None:
      fps, *_ = search(r"Frames per second\s*:\s*(\d+.\d+)", ret).groups()

    return f'{format_} {width}x{height} ({fps} fps)'

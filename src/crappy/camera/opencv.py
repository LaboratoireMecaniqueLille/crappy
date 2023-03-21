# coding: utf-8

from time import time
from typing import Tuple, List, Optional
from numpy import ndarray
from platform import system
from subprocess import run
from re import findall, split, search
import logging

from .meta_camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class CameraOpencv(Camera):
  """A class for reading images from any camera able to interface with OpenCv.

  The number of the video device to read images from can be specified. It is
  then also possible to tune the encoding format and the size.

  This camera class is less performant than the :ref:`Camera GStreamer` one
  that relies on GStreamer, but the installation of OpenCv is way easier than
  the one of GStreamer.

  Note:
    For a better performance of this class in Linux, it is recommended to have
    `v4l-utils` installed.
  """

  def __init__(self) -> None:
    """Sets variables and adds the channels setting."""

    super().__init__()

    self._cap = None
    self._device_num: Optional[int] = None
    self._formats: List[str] = list()

    self.add_choice_setting(name="channels",
                            choices=('1', '3'),
                            default='1')

  def open(self, device_num: int = 0, **kwargs) -> None:
    """Opens the video stream and sets any user-specified settings.

    Args:
      device_num (:obj:`int`, optional): The number of the device to open.
      **kwargs: Any additional setting to set before opening the graphical
        interface.
    """

    if not isinstance(device_num, int) or device_num < 0:
      raise ValueError("device_num should be an integer !")

    # Opening the videocapture device
    self.log(logging.INFO, "Opening the image stream from the camera")
    self._cap = cv2.VideoCapture(device_num)
    self._device_num = device_num
    fourcc = self._get_fourcc()

    if system() == 'Linux':
      self._formats = []

      # Trying to run v4l2-ctl to get the available formats
      command = ['v4l2-ctl', '-d', str(device_num), '--list-formats-ext']
      try:
        self.log(logging.INFO, f"Getting the available image formats with "
                               f"command {command}")
        check = run(command, capture_output=True, text=True)
      except FileNotFoundError:
        self.log(logging.WARNING, "The performance of the CameraOpencv class "
                                  "could be improved if v4l-utils was "
                                  "installed !")
        check = None
      check = check.stdout if check is not None else ''

      # Splitting the returned string to isolate each encoding
      if findall(r'\[\d+]', check):
        check = split(r'\[\d+]', check)[1:]
      elif findall(r'Pixel\sFormat', check):
        check = split(r'Pixel\sFormat', check)[1:]
      else:
        check = []

      if check:
        for img_format in check:
          # For each encoding, finding its name
          name, *_ = search(r"'(\w+)'", img_format).groups()
          sizes = findall(r'\d+x\d+', img_format)
          fps_sections = split(r'\d+x\d+', img_format)[1:]

          # For each name, finding the available sizes
          for size, fps_section in zip(sizes, fps_sections):
            fps_list = findall(r'\((\d+\.\d+)\sfps\)', fps_section)
            for fps in fps_list:
              self._formats.append(f'{name} {size} ({fps} fps)')

      else:
        # If v4l-utils is not installed, proposing two encodings without
        # further detail
        self._formats = [fourcc, 'MJPG']

        # Still letting the user choose the size
        self.add_scale_setting(name='width', lowest=1, highest=1920,
                               getter=self._get_width, setter=self._set_width)
        self.add_scale_setting(name='height', lowest=1, highest=1080,
                               getter=self._get_height,
                               setter=self._set_height)

    else:
      # On Windows the fourcc management is even messier than on Linux
      self._formats = []

      # Still letting the user choose the size
      self.add_scale_setting(name='width', lowest=1, highest=1920,
                             getter=self._get_width, setter=self._set_width)
      self.add_scale_setting(name='height', lowest=1, highest=1080,
                             getter=self._get_height, setter=self._set_height)

    if self._formats:
      # The format integrates the size selection
      if ' ' in self._formats[0]:
        self.add_choice_setting(name='format',
                                choices=tuple(self._formats),
                                getter=self._get_format_size,
                                setter=self._set_format)
      # The size is independent of the format
      else:
        self.add_choice_setting(name='format',
                                choices=tuple(self._formats),
                                getter=self._get_fourcc,
                                setter=self._set_format)

    # Setting the kwargs if any
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, ndarray]:
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
      return t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return t, frame

  def close(self) -> None:
    """Releases the videocapture object."""

    if self._cap is not None:
      self.log(logging.INFO, "Closing the image stream from the camera")
      self._cap.release()

  def _get_fourcc(self) -> str:
    """Returns the current fourcc string of the video encoding."""

    fcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
    return f"{fcc & 0xFF:c}{(fcc >> 8) & 0xFF:c}" \
           f"{(fcc >> 16) & 0xFF:c}{(fcc >> 24) & 0xFF:c}"

  def _get_width(self) -> int:
    """Returns the current image width."""

    return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_height(self) -> int:
    """Returns the current image height."""

    return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_width(self, width: int) -> None:
    """Tries to set the image width."""

    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  def _set_height(self, height: int) -> None:
    """Tries to set the image height."""

    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  def _set_format(self, img_format: str) -> None:
    """Sets the format of the image according to the user's choice."""

    # The format might be made of a name and a dimension, or just a name
    try:
      format_name, img_size, fps = findall(r"(\w+)\s(\w+)\s\((\d+.\d+) fps\)",
                                           img_format)[0]
    except ValueError:
      format_name, img_size, fps = img_format, None, None

    # Setting the format
    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format_name))

    if img_size is not None:
      # Getting the width and height from the second half of the string
      width, height = map(int, img_size.split('x'))

      # Setting the size
      self._set_width(width)
      self._set_height(height)

    if fps is not None:
      self._cap.set(cv2.CAP_PROP_FPS, float(fps))

  def _get_format_size(self) -> str:
    """Parses the v4l2-ctl -V command to get the current image format as an
    index."""

    # Sending the v4l2-ctl command
    command = ['v4l2-ctl', '-d', str(self._device_num), '--all']
    check = run(command, capture_output=True, text=True).stdout

    # Parsing the answer
    format_ = width = height = fps = ''
    if search(r"Pixel Format\s*:\s*'(\w+)'", check) is not None:
      format_, *_ = search(r"Pixel Format\s*:\s*'(\w+)'", check).groups()
    if search(r"Width/Height\s*:\s*(\d+)/(\d+)", check) is not None:
      width, height = search(r"Width/Height\s*:\s*(\d+)/(\d+)", check).groups()
    if search(r"Frames per second\s*:\s*(\d+.\d+)", check) is not None:
      fps, *_ = search(r"Frames per second\s*:\s*(\d+.\d+)", check).groups()

    return f'{format_} {width}x{height} ({fps} fps)'

# coding: utf-8

from time import time
from typing import NoReturn, Tuple
from numpy import ndarray
from platform import system
from subprocess import run
from re import findall, split
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

# Todo: Manage frame rate


class Camera_opencv(Camera):
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

    Camera.__init__(self)
    self.name = "camera_opencv"
    self._cap = None

    self.add_setting("channels", limits={1: 1, 3: 3}, default=1)

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
    self._cap = cv2.VideoCapture(device_num)
    self._device_num = device_num
    fourcc = self._get_fourcc()

    if system() == 'Linux':

      self._format_to_index = {}
      self._index_to_format = {}

      # Trying to run v4l2-ctl to get the available formats
      command = ['v4l2-ctl', '-d', str(device_num), '--list-formats-ext']
      try:
        check = run(command, capture_output=True, text=True)
      except FileNotFoundError:
        print("\n#######\n"
              "Warning ! The performance of the Camera_opencv "
              "class could be improved if v4l-utils was installed !"
              "\n#######\n")
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
          name = findall(r"'\w+'", img_format)[0].replace("'", '')
          sizes = findall(r'\d+x\d+', img_format)

          # For each name, finding the available sizes
          for size in sizes:
            self._format_to_index.update(
              {name + ' ' + size: len(self._format_to_index)})
            self._index_to_format.update(
              {len(self._index_to_format): name + ' ' + size})

      else:
        # If v4l-utils is not installed, proposing two encodings without
        # further detail
        self._format_to_index = {fourcc: 0, 'MJPG': 1}
        self._index_to_format = {0: fourcc, 1: 'MJPG'}

        # Still letting the user choose the size
        self.add_setting("width", self._get_width, self._set_width, (1, 1920))
        self.add_setting("height", self._get_height, self._set_height,
                         (1, 1080))

    else:
      # On Windows the fourcc management is even messier than on Linux
      self._format_to_index = {}

      # Still letting the user choose the size
      self.add_setting("width", self._get_width, self._set_width, (1, 1920))
      self.add_setting("height", self._get_height, self._set_height, (1, 1080))

    # Finally, creating the format parameter if applicable
    if self._format_to_index:
      # The format integrates the size selection
      if ' ' in self._index_to_format[0]:
        self.add_setting("format", self._get_format_size, self._set_format,
                         self._format_to_index)
      # The size is independent of the format
      else:
        self.add_setting("format", self._get_format_fourcc, self._set_format,
                         self._format_to_index)

    # Setting the kwargs if any, and making sure they exist
    for kwarg in kwargs:
      if kwarg not in self.available_settings:
        raise ValueError(f"Unexpected argument {kwarg} for camera "
                         f"{type(self).__name__}.")
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
    if self.channels == 1:
      return t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return t, frame

  def close(self) -> NoReturn:
    """Releases the videocapture object."""

    if self._cap is not None:
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

  def _set_width(self, width: int) -> NoReturn:
    """Tries to set the image width."""

    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  def _set_height(self, height: int) -> NoReturn:
    """Tries to set the image height."""

    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  def _set_format(self, img_format) -> NoReturn:
    """Sets the format of the image according to the user's choice."""

    # Converting the index to a string using the dict build at open()
    raw_format = self._index_to_format[img_format].split(' ')

    # isolating the name from the size
    img_format = raw_format[0]
    img_size = raw_format[1] if len(raw_format) > 1 else None

    # Setting the format
    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*img_format))

    if img_size is not None:
      # Getting the width and height from the second half of the string
      width = int(img_size.split('x')[0])
      height = int(img_size.split('x')[1])

      # Setting the size
      self._set_width(width),
      self._set_height(height)

  def _get_format_size(self) -> int:
    """Parses the v4l2-ctl -V command to get the current image format as an
    index."""

    # Sending the v4l2-ctl command
    command = ['v4l2-ctl', '-d', str(self._device_num), '-V']
    check = run(command, capture_output=True, text=True).stdout

    # Parsing the answer
    format_ = width = height = ''
    if findall(r"'\w+'", check):
      format_ = findall(r"'\w+'", check)[0].replace("'", "")
    if findall(r"\d+/\d+", check):
      width = findall(r"\d+/\d+", check)[0].split('/')[0]
      height = findall(r"\d+/\d+", check)[0].split('/')[1]

    # Returning the corresponding index
    try:
      index = self._format_to_index[format_ + ' ' + width + 'x' + height]
      return index
    except KeyError:
      raise KeyError("Couldn't retrieve the current image format.")

  def _get_format_fourcc(self) -> int:
    """Returns the current video encoding vas an index according to opencv."""

    try:
      return self._format_to_index[self._get_fourcc()]
    except KeyError:
      raise KeyError("Couldn't retrieve the current image format.")

# coding: utf-8

from __future__ import annotations
from time import time, sleep
from numpy import uint8, ndarray, uint16, copy, squeeze
from typing import Tuple, Optional, Union, List, Callable
from subprocess import Popen, PIPE, run
from re import findall, split, search, finditer, Match
import logging
from fractions import Fraction
from dataclasses import dataclass

from .meta_camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ImportError, ModuleNotFoundError):
  pass
  cv2 = OptionalModule('opencv-python')

try:
  from gi import require_version
  require_version('Gst', '1.0')
  require_version('GstApp', '1.0')
  from gi.repository import Gst, GstApp
except (ImportError, ModuleNotFoundError, ValueError):
  Gst = GstApp = OptionalModule('PyGObject')


@dataclass
class Parameter:
  """A class for the different parameters the user can adjust """

  name: str
  type: str
  min: Optional[str] = None
  max: Optional[str] = None
  step: Optional[str] = None
  default: Optional[str] = None
  value: Optional[str] = None
  flags: Optional[str] = None
  options: Optional[Tuple[str, ...]] = None

  @classmethod
  def parse_info(cls, match: Match) -> Parameter:
    """Instantiates the class Parameter, according to the information
     collected with v4l2-ctl.

    Args:
      match: Match object returned by successful matches of the regex with
      a string.

    Returns:
      The instantiated class.
    """

    return cls(name=match.group(1),
               type=match.group(2),
               min=match.group(4) if match.group(4) else None,
               max=match.group(6) if match.group(6) else None,
               step=match.group(8) if match.group(8) else None,
               default=match.group(10) if match.group(10) else None,
               value=match.group(11),
               flags=match.group(13) if match.group(13) else None)

  def add_options(self, match: Match) -> None:
    """Adds the different possible options for a menu parameter.

    Args:
      match: Match object returned by successful matches of the regex with
      a string.
    """

    menu_info = match.group(1)
    menu_values = match.group(2)
    menu_name = search(r'(\w+) \w+ \(menu\)', menu_info).group(1)
    if self.name == menu_name:
      options = findall(r'\d+: .+?(?=\n|$)', menu_values)
      self.options = tuple(options)
      for option in self.options:
        if self.default == option[0]:
          self.default = option


class CameraGstreamer(Camera):
  """A class for reading images from a video device using Gstreamer in Linux.

  It can read images from the default video source, or a video device can be
  specified. In this case, the user has access to a range of parameters for
  tuning the image. Alternatively, it is possible to give a custom GStreamer
  pipeline as an argument. In this case no settings are available, and it is up
  to the user to ensure the validity of the pipeline.

  This class uses less resources and is compatible with more cameras than the
  :class:`~crappy.camera.CameraOpencv` camera, that relies on OpenCV. The
  installation of GStreamer is however less straightforward than the one of
  OpenCV.

  To use this class, `v4l-utils` must be installed.

  Note:
    This Camera requires the module :mod:`PyGObject` to be installed, as well
    as GStreamer.
  """

  def __init__(self) -> None:
    """Simply initializes the instance attributes."""

    super().__init__()

    Gst.init(None)
    self._last_frame_nr = 0
    self._frame_nr = 0

    # These attributes will be set later
    self._pipeline = None
    self._process: Optional[Popen] = None
    self._img: Optional[ndarray] = None
    self._device: Optional[Union[str, int]] = None
    self._user_pipeline: Optional[str] = None
    self._nb_channels: int = 3
    self._img_depth: int = 8
    self._formats: List[str] = list()
    self._app_sink = None
    self.parameters = []

  def open(self,
           device: Optional[Union[int, str]] = None,
           user_pipeline: Optional[str] = None,
           nb_channels: Optional[int] = None,
           img_depth: Optional[int] = None,
           **kwargs) -> None:
    """Opens the pipeline, sets the settings and starts the acquisition of
    images.

    A custom pipeline can be specified using the ``user_pipeline`` argument.
    Simply copy-paste the pipeline as it is in the terminal, and the class will
    take care of adjusting it to make it compatible with the Python binding
    of GStreamer.

    Note:
      For custom pipelines, it may be necessary to set a format directly in the
      pipeline (which isn't the case in the terminal). For instance, this line

      .. code-block:: shell-session

        gst-launch-1.0 autovideosrc ! videoconvert ! autovideosink

      May need to become

      .. code-block:: shell-session

        gst-launch-1.0 autovideosrc ! videoconvert ! video/x-raw,format=BGR ! \
autovideosink

    Note:
      Pipelines built using a pipe for redirecting the stream from another
      application are also accepted. For example, the following user pipeline
      is valid :

      .. code-block:: shell-session

        gphoto2 --stdout --capture-movie | gst-launch-1.0 fdsrc fd=0 ! \
videoconvert ! autovideosink

    Args:
      device: The video device to open, if the device opened by default isn't
        the right one. It should be a path like `/dev/video0`. This argument
        is ignored if a ``user_pipeline`` is given.
      user_pipeline: A custom pipeline that can optionally be given as a
        :obj:`str`. If given, the ``device`` argument is ignored. The pipeline
        should be given as it would be in a terminal.
      nb_channels: The number of channels expected in the acquired images, in
        case a custom pipeline is given. Otherwise, this argument is ignored.
        For now, Crappy only manages 1- and 3-channel images.
      img_depth: The bit depth of each channel of the acquired images, in case
        a custom pipeline is given. Otherwise, this argument is ignored. For
        now, Crappy only manages 8- and 16-bits deep images.
      **kwargs: Allows specifying values for the settings even before
        displaying the configuration window.
    """

    # Checking the validity of the arguments
    if img_depth is not None and img_depth not in [8, 16]:
      raise ValueError('The img_depth must be either 8 or 16 (bits)')

    if user_pipeline is not None and nb_channels is None:
      raise ValueError('nb_channels must be given if user_pipeline is !')
    if user_pipeline is not None and img_depth is None:
      raise ValueError('img_depth must be given if user_pipeline is !')

    # Parsing the user pipeline if given
    if user_pipeline is not None:

      # Removing the call to gst-launch-1.0
      user_pipeline = user_pipeline.replace('gst-launch-1.0 ', '')
      # Splitting in case there's a pipe in the command
      user_pipeline = user_pipeline.split(' | ', 1)

      # There's no pipe in the pipeline
      if len(user_pipeline) == 1:
        user_pipeline = user_pipeline[0]

        # Setting the sink
        user_pipeline = user_pipeline.split(' ! ')
        user_pipeline[-1] = 'appsink name=sink'
        user_pipeline = ' ! '.join(user_pipeline)

      # There's a pipe in the pipeline
      else:
        user_application = user_pipeline[0]
        user_pipeline = user_pipeline[1]

        # Opening a subprocess handling the first half of the pipeline
        self.log(logging.INFO, f"Running command "
                               f"{user_application.split(' ')}")
        self._process = Popen(user_application.split(' '), stdout=PIPE)
        file_descriptor = self._process.stdout.fileno()

        # Setting the source and the sink, according to the file descriptor of
        # the subprocess
        user_pipeline = user_pipeline.split(' ! ')
        user_pipeline[0] = f'fdsrc fd={file_descriptor}'
        user_pipeline[-1] = 'appsink name=sink'
        user_pipeline = ' ! '.join(user_pipeline)

    # Setting the expected properties of the image
    self._device = device
    self._user_pipeline = user_pipeline
    self._nb_channels = nb_channels if nb_channels is not None else 3
    self._img_depth = img_depth if img_depth is not None else 8

    # Defining the settings in case no custom pipeline was given
    if user_pipeline is None:

      # Trying to get the available image encodings and formats
      self._formats = []

      # Trying to run v4l2-ctl to get the available formats
      command = ['v4l2-ctl', '--list-formats-ext'] if device is None \
          else ['v4l2-ctl', '-d', device, '--list-formats-ext']
      self.log(logging.INFO, f"Getting the available image formats with "
                             f"command {command}")
      try:
        check = run(command, capture_output=True, text=True)
      except FileNotFoundError:
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

      # Finally, creating the parameter if applicable
      if self._formats:
        if not run(['gst-inspect-1.0', 'avdec_h264'],
                   capture_output=True, text=True).stdout:
          self._formats = [form for form in self._formats
                           if form.split()[0] != 'H264']
          self.log(logging.WARNING, "The format H264 is not available"
                                    "It could be if gstreamer1.0-libav "
                                    "was installed !")
        if not run(['gst-inspect-1.0', 'avdec_h265'],
                   capture_output=True, text=True).stdout:
          self._formats = [form for form in self._formats
                           if form.split()[0] != 'HEVC']
          self.log(logging.WARNING, "The format HEVC is not available"
                                    "It could be if gstreamer1.0-libav "
                                    "was installed !")

        # The format integrates the size selection
        if ' ' in self._formats[0]:
          self.add_choice_setting(name='format',
                                  choices=tuple(self._formats),
                                  getter=self._get_format,
                                  setter=self._set_format)
        # The size is independent of the format
        else:
          self.add_choice_setting(name='format', choices=tuple(self._formats),
                                  setter=self._set_format)

      # Trying to run v4l2-ctl to get the available settings
      command = ['v4l2-ctl', '-L'] if device is None \
          else ['v4l2-ctl', '-d', device, '-L']
      self.log(logging.INFO, f"Getting the available image settings with "
                             f"command {command}")
      try:
        check = run(command, capture_output=True, text=True)
      except FileNotFoundError:
        check = None
      check = check.stdout if check is not None else ''

      # Regex to extract the different parameters and their information
      param_pattern = (r'(\w+)\s+0x\w+\s+\((\w+)\)\s+:\s*'
                       r'(min=(-?\d+)\s+)?'
                       r'(max=(-?\d+)\s+)?'
                       r'(step=(\d+)\s+)?'
                       r'(default=(-?\d+)\s+)?'
                       r'value=(-?\d+)\s*'
                       r'(flags=([^\\n]+))?')

      # Extract the different parameters and their information
      matches = finditer(param_pattern, check)
      for match in matches:
        self.parameters.append(Parameter.parse_info(match))

      # Regex to extract the different options in a menu
      menu_options = finditer(
        r'(\w+ \w+ \(menu\))([\s\S]+?)(?=\n\s*\w+ \w+ \(.+?\)|$)', check)

      # Extract the different options
      for menu_option in menu_options:
        for param in self.parameters:
          param.add_options(menu_option)

      # Create the different settings
      for param in self.parameters:
        if not param.flags:
          if param.type == 'int':
            self.add_scale_setting(name=param.name,
                                   lowest=int(param.min),
                                   highest=int(param.max),
                                   getter=self._add_scale_getter(param.name),
                                   setter=self._add_setter(param.name),
                                   default=param.default,
                                   step=int(param.step))
          elif param.type == 'bool':
            self.add_bool_setting(name=param.name,
                                  getter=self._add_bool_getter(param.name),
                                  setter=self._add_setter(param.name),
                                  default=bool(int(param.default)))
          elif param.type == 'menu':
            if param.options:
              self.add_choice_setting(name=param.name,
                                      choices=param.options,
                                      getter=self._add_menu_getter(param.name),
                                      setter=self._add_setter(param.name),
                                      default=param.default)

      self.add_choice_setting(name="channels", choices=('1', '3'), default='1')

      # Adding the software ROI selection settings
      if self._formats and ' ' in self._formats[0]:
        width, height = search(r'(\d+)x(\d+)', self._get_format()).groups()
        self.add_software_roi(int(width), int(height))

    # Setting up GStreamer and the callback
    self.log(logging.INFO, "Initializing the GST pipeline")
    self.log(logging.DEBUG, f"The pipeline is {self._get_pipeline()}")
    self._pipeline = Gst.parse_launch(self._get_pipeline())
    self._app_sink = self._pipeline.get_by_name('sink')
    self._app_sink.set_property("emit-signals", True)
    self._app_sink.connect("new-sample", self._on_new_sample)

    # Starting image acquisition
    self.log(logging.INFO, "Starting the GST pipeline")
    self._pipeline.set_state(Gst.State.PLAYING)

    # Checking that images are read as expected
    t0 = time()
    while not self._frame_nr:
      if time() - t0 > 2:
        raise TimeoutError(
          "Waited too long for the first image ! There is probably an error "
          "in the pipeline. The format of the received images may be "
          "unexpected, in which case you can try specifying it.")
      sleep(0.01)

    # Setting the kwargs if any
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, ndarray]:
    """Reads the last image acquired from the camera.

    Returns:
      The acquired image, along with a timestamp.
    """

    # Assuming an image rate greater than 0.5 FPS
    # Checking that we don't return the same image twice
    t0 = time()
    while self._last_frame_nr == self._frame_nr:
      if time() - t0 > 2:
        raise TimeoutError("Waited too long for the next image !")
      sleep(0.01)

    self._last_frame_nr = self._frame_nr

    return time(), self.apply_soft_roi(copy(self._img))

  def close(self) -> None:
    """Simply stops the image acquisition."""

    if self._pipeline is not None:
      self.log(logging.INFO, "Stopping the GST pipeline")
      self._pipeline.set_state(Gst.State.NULL)

    # Closes the subprocess started in case a user pipeline containing a pipe
    # was given
    if self._process is not None:
      self.log(logging.INFO, "Stopping the image generating process")
      self._process.terminate()

  def _restart_pipeline(self, pipeline: str) -> None:
    """Stops the current pipeline, redefines it, and restarts it.

    Args:
      pipeline: The new pipeline to use, as a :obj:`str`.
    """

    # Stops the previous pipeline
    self.log(logging.INFO, "Stopping the GST pipeline")
    self._pipeline.set_state(Gst.State.NULL)

    # Redefines the pipeline and the callbacks
    self.log(logging.INFO, "Initializing the GST pipeline")
    self.log(logging.DEBUG, f"The new pipeline is {pipeline}")
    self._pipeline = Gst.parse_launch(pipeline)
    self._app_sink = self._pipeline.get_by_name('sink')
    self._app_sink.set_property("emit-signals", True)
    self._app_sink.connect("new-sample", self._on_new_sample)

    # Restarts the pipeline
    self.log(logging.INFO, "Starting the GST pipeline")
    self._pipeline.set_state(Gst.State.PLAYING)

  def _get_pipeline(self, img_format: Optional[int] = None) -> str:
    """Method that generates a pipeline, according to the given settings.

    If a user-defined pipeline was given, it will always be returned.

    Args:
      img_format: The image format to set, as a :obj:`str` containing the name
        of the encoding and optionally both the width and height in pixels.

    Returns:
      A pipeline matching the current settings values.
    """

    # Returns the custom pipeline if any
    if self._user_pipeline is not None:
      return self._user_pipeline

    # The source argument is handled differently according to the platform
    if self._device is not None:
      device = f'device={self._device}'
    else:
      device = ''

    # Getting the format index
    img_format = img_format if img_format is not None else self.format

    try:
      format_name, img_size, fps = findall(r"(\w+)\s(\w+)\s\((\d+.\d+) fps\)",
                                           img_format)[0]
    except ValueError:
      format_name, img_size, fps = img_format, None, None

    # Adding a mjpeg decoder to the pipeline if needed
    if format_name == 'MJPG':
      img_format = '! jpegdec'
    elif format_name == 'H264':
      img_format = '! h264parse ! avdec_h264'
    elif format_name == 'HEVC':
      img_format = '! h265parse ! avdec_h265'
    elif format_name == 'YUYV':
      img_format = ''

    # Getting the width and height from the second half of the string
    if img_size is not None:
      width, height = map(int, img_size.split('x'))
    else:
      width, height = None, None

    # Including the dimensions and the fps in the pipeline
    img_size = f',width={width},height={height}' if width else ''
    if fps is not None:
      fps = Fraction(fps)
      fps_str = f',framerate={fps.numerator}/{fps.denominator}'
    else:
      fps_str = ''

    # Finally, generate a single pipeline containing all the user settings
    return f"""v4l2src {device} name=source {img_format} ! videoconvert ! 
           video/x-raw,format=BGR{img_size}{fps_str} ! appsink name=sink"""

  def _on_new_sample(self, app_sink):
    """Callback that reads every new frame and puts it into a buffer.

    Args:
      app_sink: The AppSink object containing the new frames.

    Returns:
      A GStreamer object indicating that the reading went fine.
    """

    sample = app_sink.pull_sample()
    caps = sample.get_caps()

    # Extracting the width and height info from the sample's caps
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    # Getting the actual data
    buffer = sample.get_buffer()
    # Getting read access to the buffer data
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
      raise RuntimeError("Could not map buffer data!")
    self.log(logging.DEBUG, "Grabbed new frame")

    # Casting the data into a numpy array
    try:
      numpy_frame = ndarray(
        shape=(height, width, self._nb_channels),
        dtype=uint8 if self._img_depth == 8 else uint16,
        buffer=map_info.data)
    except TypeError:
      raise TypeError("Unexpected number of channels in the received image !\n"
                      "You can try adding something like ' ! videoconvert ! "
                      "video/x-raw,format=BGR ! ' before your sink to specify "
                      "the format.\n(here BGR would be for 3 channels)")

    # Converting to gray level if needed
    if self._user_pipeline is None and self.channels == '1':
      numpy_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_BGR2GRAY)

    # Cleaning up the buffer mapping
    buffer.unmap(map_info)

    self._img = squeeze(numpy_frame)
    self._frame_nr += 1

    return Gst.FlowReturn.OK

  def _set_format(self, img_format) -> None:
    """Sets the image encoding and dimensions."""

    self._restart_pipeline(self._get_pipeline(img_format=img_format))

    # Reloading the software ROI selection settings
    if self._soft_roi_set and self._formats and ' ' in self._formats[0]:
      width, height = search(r'(\d+)x(\d+)', img_format).groups()
      self.reload_software_roi(int(width), int(height))

  def _get_format(self) -> str:
    """Parses the ``v4l2-ctl -V`` command to get the current image format as an
    index."""

    # Sending the v4l2-ctl command
    if self._device is not None:
      command = ['v4l2-ctl', '-d', str(self._device), '--all']
    else:
      command = ['v4l2-ctl', '--all']
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

  def _add_setter(self, name: str) -> Callable:
    """Creates a setter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The setter function.
    """

    def setter(value) -> None:
      """The method to set the value of a setting running v4l2-ctl.
      """

      if isinstance(value, str):
        if self._device is not None:
          command = ['v4l2-ctl', '-d', self._device, '--set-ctrl',
                     name+f'={int(value[0])}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', name+f'={int(value[0])}']
        self.log(logging.DEBUG, "Setting "+name+f" with command {command}")
        run(command, capture_output=True, text=True)
      else:
        if self._device is not None:
          command = ['v4l2-ctl', '-d', self._device, '--set-ctrl',
                     name+f'={int(value)}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', name+f'={int(value)}']
        self.log(logging.DEBUG, "Setting "+name+f" with command {command}")
        run(command, capture_output=True, text=True)
    return setter

  def _add_scale_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> int:
      """The method to get the current value of a scale setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device is not None:
        command = ['v4l2-ctl', '-d', self._device, '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, "Getting "+name+f" with command {command}")
        value = run(command, capture_output=True, text=True).stdout.split()[-1]
      except FileNotFoundError:
        value = None
      return int(value)
    return getter

  def _add_bool_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> bool:
      """The method to get the current value of a bool setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device is not None:
        command = ['v4l2-ctl', '-d', self._device, '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, "Getting " + name + f" with command {command}")
        value = run(command, capture_output=True, text=True).stdout.split()[-1]
      except FileNotFoundError:
        value = None
      return bool(int(value))
    return getter

  def _add_menu_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> str:
      """The method to get the current value of a choice setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device is not None:
        command = ['v4l2-ctl', '-d', self._device, '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, "Getting " + name + f" with command {command}")
        value = run(command, capture_output=True, text=True).stdout.split()[-1]
        for param in self.parameters:
          if param.name == name:
            for option in param.options:
              if value == option[0]:
                value = option
      except FileNotFoundError:
        value = None
      return value
    return getter

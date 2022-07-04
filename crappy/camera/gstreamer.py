# coding: utf-8

from .camera import Camera
from .._global import OptionalModule
from time import time, sleep
from numpy import uint8, ndarray, uint16, copy, squeeze
from typing import Tuple, Optional, Union
from subprocess import Popen, PIPE, run
from platform import system
from re import findall, split, search

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
  Gst = GstApp = OptionalModule('gi', 'You need to install the python bindings'
                                      ' for GStreamer ! ')

# Todo: Error with a getter


class Camera_gstreamer(Camera):
  """A class for reading images from a video device using Gstreamer.

  It can read images from the default video source, or a video device can be
  specified. In this case, the user has access to a range of parameters for
  tuning the image. Alternatively, it is possible to give a custom GStreamer
  pipeline to the block. In this case no settings are available, and it is up
  to the user to ensure the validity of the pipeline.

  This class uses less resources and is compatible with more cameras than the
  :ref:`Camera OpenCV` camera, that relies on OpenCV. The installation of
  GStreamer of however less straightforward than the one of OpenCV.

  Note:
    For a better performance of this class in Linux, it is recommended to have
    `v4l-utils` installed.
  """

  def __init__(self) -> None:
    """Simply initializes the instance attributes."""

    super().__init__()
    Gst.init(None)

    self.name = "camera_gstreamer"
    self._last_frame_nr = 0
    self._frame_nr = 0
    self._img = None

    self._exposure_mode = None

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
        the right one. In Linux, should be a path like `/dev/video0`. In
        Windows and Mac, should be the index of the video device. This argument
        is ignored if a ``user_pipeline`` is given.
      user_pipeline (:obj:`str`, optional): A custom pipeline that can
        optionally be given. If given, the ``device`` argument is ignored. The
        pipeline should be given as it would be in a terminal.
      nb_channels (:obj:`int`, optional): The number of channels expected in
        the acquired images, in case a custom pipeline is given. Otherwise,
        this argument is ignored. For now, Crappy only manages 1- and 3-channel
        images.
      img_depth (:obj:`int`, optional): The bit depth of each channel of the
        acquired images, in case a custom pipeline is given. Otherwise, this
        argument is ignored. For now, Crappy only manages 8- and 16-bits deep
        images.
      **kwargs: Allows specifying values for the settings even before
        displaying the graphical interface.
    """

    # Checking the validity of the arguments
    if device is not None and system() == 'Linux' and not isinstance(device,
                                                                     str):
      raise ValueError("In Linux, device should be a string !")
    elif device is not None and system() in ['Darwin', 'Windows'] and not \
            isinstance(device, int):
      raise ValueError("In Windows and Mac, device should be an integer !")

    if img_depth is not None and img_depth not in [8, 16]:
      raise ValueError('The img_depth must be either 8 or 16 (bits)')

    if user_pipeline is not None and nb_channels is None:
      raise ValueError('nb_channels must be given if user_pipeline is !')
    if user_pipeline is not None and img_depth is None:
      raise ValueError('img_depth must be given if user_pipeline is !')

    self._process = None

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

      # First, determine if the exposure can be set (Linux only)
      if system() == "Linux":

        # Trying to run v4l2-ctl to get the available settings
        command = ['v4l2-ctl', '-l'] if device is None \
          else ['v4l2-ctl', '-d', device, '-l']
        try:
          check = run(command, capture_output=True, text=True)
        except FileNotFoundError:
          print("\n#######\n"
                "Warning ! The performance of the Camera_gstreamer "
                "class could be improved if v4l-utils was installed !"
                "\n#######\n")
          check = None
        check = check.stdout if check is not None else ''

        # Trying to find the exposure parameters in the returned string
        expo = search(r'exposure\s0x[\dA-Fa-f]+\s\(\w+\)\s+:\s'
                      r'min=(?P<min>\d+)\smax=(?P<max>\d+)', check)
        expo_auto = search(r'exposure_auto\s0x[\dA-Fa-f]+\s\(\w+\)\s+:\s'
                           r'min=(?P<min>\d+)\smax=(?P<max>\d+)', check)
        expo_abso = search(r'exposure_absolute\s0x[\dA-Fa-f]+\s\(\w+\)\s+:\s'
                           r'min=(?P<min>\d+)\smax=(?P<max>\d+)', check)

        # If there's an exposure parameter, getting its upper and lower limits
        if expo:
          expo_min, expo_max = map(int, expo.groups())
          self._exposure_mode = 'direct'

        # If there's an exposure parameter, getting its upper and lower limits
        elif expo_auto and expo_abso:
          expo_min, expo_max = map(int, expo_abso.groups())
          self._exposure_mode = 'auto'

        else:
          expo_min = None
          expo_max = None

        # Creating the parameter if applicable
        if expo_min is not None:
          self.add_scale_setting(name='exposure', lowest=expo_min,
                                 highest=expo_max, getter=self._get_exposure,
                                 setter=self._set_exposure)

      # Then, trying to get the available image encodings and formats
      if system() == 'Linux':

        self._formats = []

        # Trying to run v4l2-ctl to get the available formats
        command = ['v4l2-ctl', '--list-formats-ext'] if device is None \
            else ['v4l2-ctl', '-d', device, '--list-formats-ext']
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

            # For each name, finding the available sizes
            for size in sizes:
              self._formats.append(f'{name} {size}')

        else:
          # If v4l-utils is not installed, proposing two encodings without
          # further detail
          self._formats = ['Default', 'MJPG']

      # For Windows and Mac proposing two encodings without further detail
      else:
        self._formats = ['Default', 'MJPG']

      # Finally, creating the parameter if applicable
      if self._formats:
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

      # These settings are always available no matter the platform
      self.add_choice_setting(name="channels", choices=('1', '3'), default='1')
      self.add_scale_setting(name='brightness', lowest=-1., highest=1.,
                             setter=self._set_brightness, default=0.)
      self.add_scale_setting(name='contrast', lowest=0., highest=2.,
                             setter=self._set_contrast, default=1.)
      self.add_scale_setting(name='hue', lowest=-1., highest=1.,
                             setter=self._set_hue, default=0.)
      self.add_scale_setting(name='saturation', lowest=0., highest=2.,
                             setter=self._set_saturation, default=1.)

    # Setting up GStreamer and the callback
    self._pipeline = Gst.parse_launch(self._get_pipeline())
    self._app_sink = self._pipeline.get_by_name('sink')
    self._app_sink.set_property("emit-signals", True)
    self._app_sink.connect("new-sample", self._on_new_sample)

    # Starting image acquisition
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

    return time(), copy(self._img)

  def close(self) -> None:
    """Simply stops the image acquisition."""

    self._pipeline.set_state(Gst.State.NULL)

    # Closes the subprocess started in case a user pipeline containing a pipe
    # was given
    if self._process is not None:
      self._process.terminate()

  def _restart_pipeline(self,
                        pipeline: str,
                        exposure: Optional[int] = None) -> None:
    """Stops the current pipeline, redefines it, and restarts it.

    Args:
      pipeline (:obj:`str`): The new pipeline to use.
      exposure (:obj:`int`, optional): The new exposure value to set.
    """

    # Stops the previous pipeline
    self._pipeline.set_state(Gst.State.NULL)

    # Redefines the pipeline and the callbacks
    self._pipeline = Gst.parse_launch(pipeline)
    self._app_sink = self._pipeline.get_by_name('sink')
    self._app_sink.set_property("emit-signals", True)
    self._app_sink.connect("new-sample", self._on_new_sample)

    # There's an extra step to set the exposure
    if system() == "Linux" and exposure is not None and \
            self._exposure_mode is not None:
      # Two different ways of setting the exposure
      if self._exposure_mode == 'direct':
        extra_args = (f"controls,"
                      f"""exposure={exposure if exposure is not None 
                                else self.exposure}""")
      else:
        extra_args = (f"controls,exposure_auto=1,"
                      f"""exposure_absolute={exposure if exposure is not None
                                         else self.exposure}""")

      # Setting the exposure is done via the 'extra-controls' property of the
      # 'v4l2src' source
      self._app_source = self._pipeline.get_by_name('source')
      structure, _ = Gst.structure_from_string(extra_args)
      self._app_source.set_property('extra-controls', structure)

    # Restarts the pipeline
    self._pipeline.set_state(Gst.State.PLAYING)

  def _get_pipeline(self,
                    brightness: Optional[float] = None,
                    contrast: Optional[float] = None,
                    hue: Optional[float] = None,
                    saturation: Optional[float] = None,
                    img_format: Optional[int] = None) -> str:
    """Method that generates a pipeline, according to the settings.

    If a user-defined pipeline was given, it will always be returned.

    Args:
      brightness: The brightness value to set, as a float between -1 and 1.
      contrast: The contrast value to set, as a float between 0 and 2.
      hue: The hue value to set, as a float between -1 and 1.
      saturation: The saturation value to set, as a float between 0 and 2.
      img_format: The image format to set, as a string containing the name of
        the encoding and optionally both the width and height in pixels.

    Returns:
      A pipeline matching the current settings values.
    """

    # Returns the custom pipeline if any
    if self._user_pipeline is not None:
      return self._user_pipeline

    # Choosing the source according to the platform
    if system() == "Linux":
      source = 'v4l2src'
    elif system() == 'Windows':
      source = 'ksvideosrc'
    elif system() == 'Darwin':
      source = 'vfsvideosrc'
    else:
      source = 'autovideosrc'

    # The source argument is handled differently according to the platform
    if system() == 'Linux' and self._device is not None:
      device = f'device={self._device}'
    elif system() in ['Darwin', 'Windows'] and self._device is not None:
      device = f'device-index={self._device}'
    else:
      device = ''

    # Getting the format index
    img_format = img_format if img_format is not None else self.format

    try:
      format_name, img_size = img_format.split(' ')
    except ValueError:
      format_name, img_size = img_format, None

    # Adding a mjpeg decoder to the pipeline if needed
    img_format = '! jpegdec' if format_name == 'MJPG' else ''

    # Getting the width and height from the second half of the string
    if img_size is not None:
      width, height = map(int, img_size.split('x'))
    else:
      width, height = None, None

    # Including the dimensions in the pipeline
    img_size = f',width={width},height={height}' if width else ''

    # Finally, generate a single pipeline containing all the user settings
    return f"""{source} {device} name=source {img_format} ! videoconvert ! 
           video/x-raw,format=BGR{img_size} ! 
           videobalance 
           brightness={brightness if brightness is not None 
                       else self.brightness:.3f} 
           contrast={contrast if contrast is not None else self.contrast:.3f} 
           hue={hue if hue is not None else self.hue:.3f}
           saturation={saturation if saturation is not None 
                       else self.saturation:.3f} ! 
           appsink name=sink"""

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

  def _set_brightness(self, brightness) -> None:
    """Sets the image brightness."""

    self._restart_pipeline(self._get_pipeline(brightness=brightness))

  def _set_contrast(self, contrast) -> None:
    """Sets the image contrast."""

    self._restart_pipeline(self._get_pipeline(contrast=contrast))

  def _set_hue(self, hue) -> None:
    """Sets the image hue."""

    self._restart_pipeline(self._get_pipeline(hue=hue))

  def _set_saturation(self, saturation) -> None:
    """Sets the image saturation."""

    self._restart_pipeline(self._get_pipeline(saturation=saturation))

  def _set_exposure(self, exposure: int) -> None:
    """Sets the exposure using v4l2.

    Only works when the platform is Linux.
    """

    self._restart_pipeline(self._get_pipeline(), exposure=exposure)

  def _get_exposure(self) -> int:
    """Returns the current exposure value.

    Only works when the platform is Linux.
    """

    # Trying to run v4l2-ctl to get the exposure value
    if self._device is not None:
      command = ['v4l2-ctl', '-d', self._device, '-C',
                 'exposure_absolute' if self._exposure_mode == 'auto' else
                 'exposure']
    else:
      command = ['v4l2-ctl', '-C',
                 'exposure_absolute' if self._exposure_mode == 'auto' else
                 'exposure']
    try:
      expo = run(command, capture_output=True, text=True)
    except FileNotFoundError:
      expo = None

    # Extracting the exposure from the returned string
    expo = expo.stdout if expo is not None else ''
    expo = search(r'\d+', expo)
    if expo:
      return int(expo[0])
    else:
      raise IOError("Couldn't read exposure value from v4l2 !")

  def _set_format(self, img_format) -> None:
    """Sets the image encoding and dimensions."""

    self._restart_pipeline(self._get_pipeline(img_format=img_format))

  def _get_format(self) -> str:
    """Parses the v4l2-ctl -V command to get the current image format as an
    index."""

    # Sending the v4l2-ctl command
    if self._device is not None:
      command = ['v4l2-ctl', '-d', str(self._device), '-V']
    else:
      command = ['v4l2-ctl', '-V']
    check = run(command, capture_output=True, text=True).stdout

    # Parsing the answer
    format_ = width = height = ''
    if search(r"'(\w+)'", check) is not None:
      format_, *_ = search(r"'(\w+)'", check).groups()
    if search(r"(\d+)/(\d+)", check):
      width, height = search(r"(\d+)/(\d+)", check).groups()

    return f'{format_} {width}x{height}'

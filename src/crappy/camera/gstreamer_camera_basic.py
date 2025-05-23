# coding: utf-8

from time import time, sleep
from numpy import uint8, ndarray, uint16, copy, squeeze
from typing import Optional, Union
from re import findall, search
from subprocess import Popen, PIPE, run
from fractions import Fraction
from platform import system
import logging

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


class CameraGstreamer(Camera):
  """A class for reading images from a video device using Gstreamer.

  It can read images from the default video source, or a video device can be
  specified. The user has access to a range of parameters for tuning the image,
  depending on the capability of the hardware. Alternatively, it is possible to
  give a custom GStreamer pipeline as an argument. In this case no settings are
  available, and it is up to the user to ensure the validity of the pipeline.

  This class uses less resources and is compatible with more cameras than the
  :class:`~crappy.camera.CameraOpencv` camera, that relies on OpenCV. The
  installation of GStreamer is however less straightforward than the one of
  OpenCV, especially on Windows !

  Warning:
    There are two classes for CameraGstreamer, one for Linux based on
    `v4l-utils`, and another one for Linux (without `v4l-utils`) and other OS.
    Depending on the installation of `v4l-utils` and the OS, the correct class
    will be automatically imported. The version using `v4l-utils` allows tuning
    more parameters than the basic version.

  Note:
    This Camera requires the module :mod:`PyGObject` to be installed, as well
    as GStreamer.
  
  .. versionadded:: 1.5.9
  .. versionchanged:: 2.0.0
     renamed from *Camera_gstreamer* to *CameraGstreamer*
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
    self._formats: list[str] = list()
    self._app_sink = None

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

      # Checking the available formats with the Gst device monitor
      cam = None
      device_monitor = Gst.DeviceMonitor.new()
      device_monitor.start()
      devices = [device for device in device_monitor.get_devices()
                 if device.get_device_class() == "Video/Source"]

      # Stop right here if there is no connected camera
      if not devices:
        self.log(logging.ERROR, "No camera devices available for reading !")
        raise IOError

      # Finding the specified device in the available ones
      if self._device is not None:
        for device in devices:
          properties = device.get_properties()
          api = properties.get_value('device.api')
          dev = properties.get_value('object.path')
          if dev == f'{api}:{self._device}':
            cam = device
            break

      # Raising an exception if the specified device cannot be found
      if self._device is not None and cam is None:
        self.log(logging.ERROR, f"Could not open the specified camera "
                                f"at: {self._device}")
        raise IOError

      # Defining a default camera if the device is not specified
      if self._device is None:
        cam = devices[0]
        path = cam.get_properties().get_value('object.path')
        self.log(logging.WARNING, f"No specific camera to open specified, "
                                  f"opening {path.split(':')[1]} by default")

      # Getting the available image formats
      caps = cam.get_caps()
      nb_formats = caps.get_size()
      for i in range(nb_formats):
        form = None
        struct = caps.get_structure(i)
        name = struct.get_name()
        if name == 'image/jpeg':
          form = 'MJPG'
        elif name == 'video/x-h264':
          form = 'H264'
        elif name == 'video/x-hevc':
          form = 'HEVC'
        elif name == 'video/x-raw':
          form = struct.get_value('format')
        else:
          self.log(logging.WARNING, f"Found unsupported cap format: {name}, "
                                    f"ignoring it")

        # Formatting the format into the v4l2 syntax
        if form is not None:
          width = struct.get_value('width')
          height = struct.get_value('height')
          framerates = findall(r'(\d+/\d+)', struct.to_string())
          for fps_str in framerates:
            fps = f"{float(Fraction(fps_str)):.3f}"
            self._formats.append(f"{form} {width}x{height} ({fps} fps)")

      device_monitor.stop()

      # Checking if the formats are supported with the installed libraries
      if self._formats:
        h264_available = bool(run(['gst-inspect-1.0', 'avdec_h264'],
                                  capture_output=True, text=True).stdout)
        h265_available = bool(run(['gst-inspect-1.0', 'avdec_h265'],
                                  capture_output=True, text=True).stdout)
        if not h264_available and any(form.split()[0] == 'H264'
                                      for form in self._formats):
          self._formats = [form for form in self._formats
                           if form.split()[0] != 'H264']
          self.log(logging.WARNING, "The video format H264 could be available "
                                    "for the selected camera if "
                                    "gstreamer1.0-libav was installed !")
        if not h265_available and any(form.split()[0] == 'HEVC'
                                      for form in self._formats):
          self._formats = [form for form in self._formats
                           if form.split()[0] != 'HEVC']
          self.log(logging.WARNING, "The video format H265 could be available "
                                    "for the selected camera if "
                                    "gstreamer1.0-libav was installed !")

      if self._formats:
        # Creating the format parameter
        self.add_choice_setting(name='format',
                                choices=tuple(self._formats),
                                setter=self._set_format)

      # These settings are always available no matter the platform
      if self._nb_channels > 1:
        self.add_choice_setting(name="channels", choices=('1', '3'),
                                default='1')
      self.add_scale_setting(name='brightness', lowest=-1., highest=1.,
                             setter=self._set_brightness, default=0.)
      self.add_scale_setting(name='contrast', lowest=0., highest=2.,
                             setter=self._set_contrast, default=1.)
      self.add_scale_setting(name='hue', lowest=-1., highest=1.,
                             setter=self._set_hue, default=0.)
      self.add_scale_setting(name='saturation', lowest=0., highest=2.,
                             setter=self._set_saturation, default=1.)

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

  def get_image(self) -> tuple[float, ndarray]:
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

  def _get_pipeline(self,
                    brightness: Optional[float] = None,
                    contrast: Optional[float] = None,
                    hue: Optional[float] = None,
                    saturation: Optional[float] = None,
                    img_format: Optional[str] = None) -> str:
    """Method that generates a pipeline, according to the given settings.

    If a user-defined pipeline was given, it will always be returned.

    Args:
      brightness: The brightness value to set, as a :obj:`float` between -1 and
        1.
      contrast: The contrast value to set, as a :obj:`float` between 0 and 2.
      hue: The hue value to set, as a :obj:`float` between -1 and 1.
      saturation: The saturation value to set, as a :obj:`float` between 0 and
        2.
      img_format: The image format to set, as a :obj:`str` containing the name
        of the encoding and optionally both the width and height in pixels.

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

    # Getting the format string if not provided as an argument
    if img_format is None and hasattr(self, 'format'):
      img_format = self.format
    # In case it goes wrong, just try without specifying the format
    if img_format is None:
      img_format = ''

    try:
      format_name, img_size, fps = findall(r"(\w+)\s(\w+)\s\((\d+.\d+) fps\)",
                                           img_format)[0]
    except ValueError:
      format_name, img_size, fps = img_format, None, None

    # Default color is BGR in most cases
    color = 'BGR'
    # Adding the decoder to the pipeline if needed
    if format_name == 'MJPG':
      img_format = '! jpegdec'
    elif format_name == 'H264':
      img_format = '! h264parse ! avdec_h264'
    elif format_name == 'HEVC':
      img_format = '! h265parse ! avdec_h265'
    elif format_name in ('YUYV', 'YUY2'):
      img_format = ''
    elif format_name == 'GRAY8':
      img_format = ''
      color = 'GRAY8'
    else:
      self.log(logging.WARNING, f"Unsupported format name: {format_name}, "
                                f"trying without explicitly setting the "
                                f"decoder in the pipeline")
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
    return f"""{source} {device} name=source {img_format} ! videoconvert ! 
           video/x-raw,format={color}{img_size}{fps_str} ! videobalance 
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
    if (self._user_pipeline is None
        and hasattr(self, 'channels')
        and self.channels == '1'):
      numpy_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_BGR2GRAY)

    # Cleaning up the buffer mapping
    buffer.unmap(map_info)

    self._img = squeeze(numpy_frame)
    self._frame_nr += 1

    return Gst.FlowReturn.OK

  def _set_brightness(self, brightness: float) -> None:
    """Sets the image brightness."""

    self._restart_pipeline(self._get_pipeline(brightness=brightness))

  def _set_contrast(self, contrast: float) -> None:
    """Sets the image contrast."""

    self._restart_pipeline(self._get_pipeline(contrast=contrast))

  def _set_hue(self, hue: float) -> None:
    """Sets the image hue."""

    self._restart_pipeline(self._get_pipeline(hue=hue))

  def _set_saturation(self, saturation: float) -> None:
    """Sets the image saturation."""

    self._restart_pipeline(self._get_pipeline(saturation=saturation))

  def _set_format(self, img_format) -> None:
    """Sets the image encoding and dimensions."""

    self._restart_pipeline(self._get_pipeline(img_format=img_format))

    # Reloading the software ROI selection settings
    if self._soft_roi_set and self._formats and ' ' in self._formats[0]:
      width, height = search(r'(\d+)x(\d+)', img_format).groups()
      self.reload_software_roi(int(width), int(height))

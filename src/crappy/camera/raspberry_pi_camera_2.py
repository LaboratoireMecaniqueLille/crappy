# coding: utf-8

import numpy as np
from typing import Optional, Any, Union, Literal
import logging
from dataclasses import dataclass
from time import time, strftime, gmtime
from threading import RLock

from .meta_camera import Camera
from .._global import OptionalModule

try:
  from picamera2 import Picamera2
  from picamera2.encoders import Encoder
  from picamera2.outputs import Output
  from picamera2.request import MappedArray, CompletedRequest
  from picamera2.controls import ControlType
except (ModuleNotFoundError, ImportError):
  Picamera2 = OptionalModule("picamera2")
  Encoder = OptionalModule("picamera2")
  Output = OptionalModule("picamera2")
  MappedArray = OptionalModule("picamera2")
  CompletedRequest = OptionalModule("picamera2")

try:
  from libcamera import controls
  # Mapping of all the autofocus modes
  AUTO_FOCUS_MODE = {
    controls.AfModeEnum.Continuous: 'Continuous',
    controls.AfModeEnum.Manual: 'Manual',
    controls.AfModeEnum.Auto: 'Auto'}
  AUTO_FOCUS_MODE_INV = {
    val: key for key, val in AUTO_FOCUS_MODE.items()}
except (ModuleNotFoundError, ImportError):
  controls = OptionalModule("libcamera")
  AUTO_FOCUS_MODE = OptionalModule("libcamera")
  AUTO_FOCUS_MODE_INV = OptionalModule("libcamera")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

# Pixel formats implemented in Crappy
PIXEL_FORMATS = ('YUV420', 'RGB888')

# Protect import to avoid raising exception when picamera2 is not installed
if not isinstance(Encoder, OptionalModule):


  class CrappyEncoder(Encoder):
    """Overloading of the :class:`picamera2.Encoder` class for the specific
    pipeline implemented in Crappy.

    Compared to the original class, passes a mapped array to the Output instead
    of a frame buffer, and also transmits all the metadata instead of just the
    timestamp.
    """

    def outputframe(self,
                    array: MappedArray,
                    metadata: dict[str, Any]) -> None:
      """Passes the mapped array of the captured frame and its metadata to the
      Output object."""

      # _output_lock might not be available depending on the version of
      # picamera2
      if hasattr(self, '_output_lock'):
        with self._output_lock:
          for out in self._output:
            out.outputframe(array, metadata)
      else:
        for out in self._output:
          out.outputframe(array, metadata)

    def _encode(self, stream, request: CompletedRequest) -> None:
      """Converts a captured image to a numpy array, extracts its metadata, and
      passes them to the Output."""

      # Probably useless here, but included for consistency with original
      # method
      if isinstance(stream, str):
        stream = request.stream_map[stream]

      # Gets the metadata and passes the frame as a mapped array
      metadata = request.get_metadata()
      with MappedArray(request, stream, reshape=True, write=True) as array:
        self.outputframe(array, metadata)


# Protect import to avoid raising exception when picamera2 is not installed
if not isinstance(Output, OptionalModule):


  class CrappyOutput(Output):
    """Overloading of the :class:`picamera2.Encoder` class for the specific
    pipeline implemented in Crappy.

    Each time a frame is received, shares it with the Camera object along with
    its metadata.
    """

    def __init__(self,
                 shared: dict[str, Union[Optional[np.ndarray],
                                         dict[str, Any]]],
                 lock: RLock) -> None:
      """Initializes the parent class and sets the arguments.

      Args:
        shared: a :obj:`dict` used for sharing the acquired frames and their
          metadata with the Camera object.
        lock: An :obj:`~threading.RLock` ensuring the Output and the Camera
          objects are not reading/writing in the shared :obj:`dict` at the same
          time.
      """

      super().__init__(pts=None)

      self._shared: dict[str, Union[Optional[np.ndarray],
                                    dict[str, Any]]] = shared
      self._frame_count: int = 0
      self._lock = lock

    def outputframe(self, array: MappedArray,
                    metadata: dict[str, Any]) -> None:
      """Shares the acquired frame and part of its metadata with the Camera
      object."""

      # Specify a limited set of metadata fields we're interested in
      to_retrieve = ('SensorTimestamp', 'ExposureTime',
                     'AnalogueGain', 'DigitalGain')

      with self._lock:
        # Place the acquired image in the shared dict
        self._shared['array'] = array.array.copy()
        # Place a subset of the metadata fields in the shared dict
        self._shared['metadata'] = {key: val for key, val in metadata.items()
                                    if key in to_retrieve and key in metadata}

        # Add a few extra fields to the metadata dictionary
        t = time()
        self._shared['metadata'] |= {'ImageUniqueID': self._frame_count,
                                     't(s)': t,
                                     'DateTimeOriginal':
                                     strftime("%Y:%m:%d %H:%M:%S", gmtime(t)),
                                     'SubsecTimeOriginal': f'{t % 1:.6f}'}
        self._frame_count += 1


@dataclass
class SensorMode:
  """Structure containing all the information about a sensor mode supported by
  the used camera."""
  
  format: str
  unpacked: str
  bit_depth: int
  size: tuple[int, int]
  fps: float
  crop_limits: tuple[int, int, int, int]
  exposure_limits: tuple[int, ...]
  
  @property
  def name(self) -> str:
    """Convenience property returning the name of the sensor mode, as displayed
    in the configuration window."""
    
    return (f"{self.size[0]}x{self.size[1]}, "
            f"{self.bit_depth}bits, {self.fps:.0f}fps")
    

class RaspberryPiCamera2(Camera):
  """:class:`~crappy.camera.Camera` object reading images from Raspberry Pi
  camera hardware, using the :mod:`picamera2` module.

  It is designed to interface seamlessly with any official Raspberry Pi camera
  module, and the other unofficial camera modules supported by
  :mod:`picamera2`. It can read images in color or grey level, in any of the
  video modes supported by the camera. Only the main camera settings are
  exposed, the more specific ones have been left out.

  Note:
    This class was only tested on PiCameraV3 and PiCameraHQ hardware, using a
    Raspberry Pi 5.

  Important:
    This class interfaces with the same hardware as
    :class:`~crappy.camera.RaspberryPiCamera`, but using an updated library.
    It is strongly recommended to use this class instead of the legacy one.

  .. versionadded:: 2.0.7
  """
  
  def __init__(self) -> None:
    """Sets the attributes and initializes the parent class."""
    
    super().__init__()
    
    # Make sure there are cameras available to read
    if not (available := Picamera2.global_camera_info()):
      raise RuntimeError("No camera detected by picamera2, aborting !")
    
    # Display the available cameras in debug message
    available_msg = '\n'.join([f"camera n° {cam['Num']}: {cam['Model']}" 
                               for cam in available])
    self.log(logging.DEBUG, f"Available cameras:{available_msg}")
    
    # Initialize objects to be used later
    self._cam: Optional[Picamera2] = None
    self._sensor_modes: Optional[list[SensorMode]] = None
    self._encoder: Optional[Encoder] = None
    self._output: Optional[Output] = None
    self._current_sensor_mode: Optional[SensorMode] = None
    self._started: bool = False

    # Initialize the objects used for sharing data with the Output
    self._shared: dict[str, Union[Optional[np.ndarray], dict[str, Any]]] = {
        'array': None,
        'metadata': dict()}
    self._lock: RLock = RLock()
    self._last_id: int = -1
  
  def open(self,
           camera_num: int = 0,
           sensor_mode: Optional[str] = None,
           pixel_format: Optional[Literal['YUV420', 'RGB888']] = None,
           grey_level_images: Optional[bool] = None,
           auto_exposure: Optional[bool] = None,
           auto_focus: Optional[Literal['Auto', 'Manual', 'Off']] = None,
           analog_gain: Optional[float] = None,
           auto_white_balance: Optional[bool] = None,
           brightness: Optional[float] = None,
           contrast: Optional[float] = None,
           exposure_time: Optional[int] = None,
           auto_exposure_value: Optional[float] = None,
           lens_position: Optional[float] = None,
           saturation: Optional[float] = None,
           sharpness: Optional[float] = None,
           soft_roi_width: Optional[int] = None,
           soft_roi_height: Optional[int] = None,
           soft_roi_x: Optional[int] = None,
           soft_roi_y: Optional[int] = None) -> None:
    """Opens the connection to the camera, instantiates the available settings
    and starts the acquisition.

    Also sets custom values for the settings if provided by the user, otherwise
    sets them to their default.

    Args:
      camera_num: An :obj:`int` specifying the number of the camera to read, if
        several cameras are connected.
      sensor_mode: The sensor mode to use for image acquisition, as a
        :obj:`str`. The available options depend on the used hardware, and will
        have the format: ``'<width>x<height>, <framerate>fps``.
      pixel_format: The pixel format to use for image acquisition. Should be
        one of:
        ::

          'YUV420', 'RGB888'

        No major difference was observed during tests between these two modes.
        The RGB mode should be preferred.
      grey_level_images: A :obj:`bool` indicating whether to convert the
        acquired images to gray level. The images are always acquired in color.
      auto_exposure: A :obj:`bool` indicating whether auto-exposure should be
        enabled, if supported by the camera.
      auto_focus: The autofocus mode to use if supported by the camera, as a
        :obj:`str`. Should be one of:
        ::

          'Auto', 'Manual', 'Off'

      analog_gain: The analog gain to use for image acquisition if supported by
        the camera, as a :obj:`float`. The possible values depend on the used
        camera.
      auto_white_balance: A :obj:`bool` indicating whether to activate the auto
        white balance, if supported by the camera.
      brightness: The brightness correction to bring to the acquired images, as
        a :obj:`float` between `-1.0` and `1.0`.
      contrast: The contrast correction to bring to the acquired images, as
        a :obj:`float` between `0.0` and `32.0`.
      exposure_time: The exposure time for image acquisition in microseconds,
        as an :obj:`int` between `5000` and `1000000`. The limits set in Crappy
        correspond to usual settings on experimental setups, but are much
        narrower than the actual sensor limits.
      auto_exposure_value: A parameter allowing to adjust the exposure time
        when using auto-exposure, as a :obj:`float` between `-8.0` and `8.0`.
      lens_position: If supported by the camera, a :obj:`float` that sets the
        position of the motorized lens and allows adjusting the focus. The
        possible values depend on the used camera.
      saturation: The saturation correction to bring to the acquired images, as
        a :obj:`float` between `0.0` and `32.0`.
      sharpness: The sharpness correction to bring to the acquired images, as
        a :obj:`float` between `0.0` and `16.0`.
      soft_roi_width: The maximum width of the cropped image, when applying a
        software ROI.
      soft_roi_height: The maximum height of the cropped image, when applying a
        software ROI.
      soft_roi_x: The `x` offset of the cropped image on the original acquired
        frame, when applying a software ROI.
      soft_roi_y: The `y` offset of the cropped image on the original acquired
        frame, when applying a software ROI.
    """
    
    # Opening the chosen camera
    self.log(logging.DEBUG, f"Opening the camera n°{camera_num}")
    self._cam = Picamera2(camera_num=camera_num)
    available_controls = self._cam.camera_controls

    # Listing all the available sensor modes for the opened camera
    self._sensor_modes = [SensorMode(mode['format'],
                                     mode['unpacked'],
                                     mode['bit_depth'],
                                     mode['size'],
                                     mode['fps'],
                                     mode['crop_limits'],
                                     mode['exposure_limits']) 
                          for mode in self._cam.sensor_modes]
    self.log(logging.DEBUG, f"Available sensor modes: {self._sensor_modes}")

    # Characteristics of the current sensor mode are needed later
    self._current_sensor_mode = self._get_mode_from_str(
        self._get_sensor_mode())
    
    # These settings always have the same value
    self._cam.video_configuration.use_case = 'video'
    self._cam.video_configuration.buffer_count = 5
    self._cam.video_configuration.display = None
    self._cam.video_configuration.encode = 'main'
    self._cam.video_configuration.queue = True
    self._cam.video_configuration.controls['FrameDurationLimits'] = (2500,
                                                                     2000000)

    # Hardware mode parameters
    self.add_choice_setting('sensor_mode',
                            [mode.name for mode in self._sensor_modes],
                            self._get_sensor_mode,
                            self._set_sensor_mode,
                            self._get_sensor_mode())
    self.add_choice_setting('pixel_format',
                            PIXEL_FORMATS,
                            self._get_pixel_format,
                            self._set_pixel_format,
                            self._get_pixel_format())

    # Auto white balance parameters
    if 'AwbEnable' in available_controls:
      self.add_bool_setting('auto_white_balance', 
                            None,
                            self._set_auto_white_balance, 
                            False)

    # Generic camera parameters that should be available most of the time
    if 'AnalogueGain' in available_controls:
      self.add_scale_setting('analog_gain',
                             available_controls['AnalogueGain'][0],
                             available_controls['AnalogueGain'][1],
                             None,
                             self._set_gain,
                             available_controls['AnalogueGain'][0])
    if 'Brightness' in available_controls:
      self.add_scale_setting('brightness', 
                             -1.0, 
                             1.0, 
                             None, 
                             self._set_brightness, 
                             0.0)
    if 'Contrast' in available_controls:
      self.add_scale_setting('contrast', 
                             0.0, 
                             32.0, 
                             None, 
                             self._set_contrast, 
                             1.0)
    if 'Saturation' in available_controls:
      self.add_scale_setting('saturation',
                             0.0,
                             32.0,
                             None,
                             self._set_saturation,
                             1.0)
    if 'Sharpness' in available_controls:
      self.add_scale_setting('sharpness',
                             0.0,
                             16.0,
                             None,
                             self._set_sharpness,
                             1.0)

    # Exposure related parameters
    if 'AeEnable' in available_controls:
      self.add_bool_setting('auto_exposure',
                            None,
                            self._set_auto_exposure,
                            False)
    if 'ExposureTime' in available_controls:
      low = self._current_sensor_mode.exposure_limits[0]
      high = self._current_sensor_mode.exposure_limits[1]
      self.add_scale_setting('exposure_time', 
                             max(low, 5000),
                             min(high, 1000000), 
                             None,
                             self._set_exposure_time, 
                             min(max(33333, low), high))
    if 'ExposureValue' in available_controls:
      self.add_scale_setting('auto_exposure_value', 
                             -8.0, 
                             8.0, 
                             None, 
                             self._set_auto_exposure_value, 
                             0.0)

    # Focus related parameters
    if 'AfMode' in available_controls:
      self.add_choice_setting('auto_focus',
                              AUTO_FOCUS_MODE.values(),
                              None,
                              self._set_auto_focus,
                              'Manual')
    if 'LensPosition' in available_controls:
      self.add_scale_setting('lens_position', 
                             available_controls['LensPosition'][0], 
                             available_controls['LensPosition'][1],
                             None, 
                             self._set_lens_position, 
                             available_controls['LensPosition'][0])

    # Parameters driving the post-processing performed in Crappy
    self.add_bool_setting('grey_level_images',
                          None,
                          None,
                          False)
    self.add_software_roi(self._current_sensor_mode.size[0],
                          self._current_sensor_mode.size[1])
    
    # Collecting the kwargs to set and setting them
    to_set = {name: arg for name, arg in zip(
        ('sensor_mode', 'pixel_format', 'grey_level_images', 'auto_exposure',
         'auto_focus', 'analog_gain', 'ROI_width', 'ROI_height', 'ROI_x',
         'ROI_y', 'auto_white_balance', 'brightness', 'contrast', 
         'exposure_time', 'auto_exposure_value', 'lens_position', 'saturation', 
         'sharpness'),
        (sensor_mode, pixel_format, grey_level_images, auto_exposure,
         auto_focus, analog_gain, soft_roi_width, soft_roi_height, soft_roi_x,
         soft_roi_y, auto_white_balance, brightness, contrast, exposure_time, 
         auto_exposure_value, lens_position, saturation, sharpness))
              if arg is not None and name in self.settings}
    self.set_all(**to_set)

    # Applying the video configuration
    self.log(logging.DEBUG, "Applying the video configuration on the camera")
    self._cam.configure('video')

    # Defining custom Encoder and Output objects
    self._encoder = CrappyEncoder()
    self._output = CrappyOutput(shared=self._shared, lock = self._lock)

    # Starting the image acquisition
    self.log(logging.DEBUG, "Starting image acquisition")
    self._cam.start_recording(self._encoder, self._output)
    self._started = True
  
  def get_image(self) -> Optional[tuple[dict[str, Any], np.ndarray]]:
    """Grabs the latest image from the shared buffer, converts it to RGB and
    grey level if necessary, and returns it along with its metadata."""

    # Accessing data using a lock to avoid simultaneous access
    with self._lock:
      # Only proceeding if there is a new image in the shared buffer
      if (self._shared['array'] is not None
          and self._shared['metadata'] 
          and self._shared['metadata']['ImageUniqueID'] > self._last_id):
        self._last_id = self._shared['metadata']['ImageUniqueID']

        # Dirty trick to avoid communication with hardware in this critical
        # section
        # Also, checking the shape of the array because some frames from the
        # previous setting might still be in the buffer
        if (self.settings['pixel_format']._value_no_getter == 'YUV420' 
            and len(self._shared['array'].shape) == 2):
          # For YUV images, converting to RGB
          img = cv2.cvtColor(self._shared['array'], cv2.COLOR_YUV420p2RGB)
        else:
          img = self._shared['array']

        # Converting the grey level if requested by the user
        if self.settings['grey_level_images'].value:
          img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return self._shared['metadata'], img
  
  def close(self) -> None:
    """Stops the image acquisition and closes the connection to the camera."""
    
    if self._cam is not None:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()
      self.log(logging.DEBUG, "Closing the connection to the camera")
      self._cam.close()
  
  def _set_sensor_mode(self, mode_str: str) -> None:
    """"""

    # Get the SensorMode object corresponding to the requested mode
    mode = [mode for mode in self._sensor_modes if mode.name == mode_str][0]
    self._current_sensor_mode = mode
    pixel_format = self._get_pixel_format()

    # The sensor mode cannot be adjusted while the camera is running
    if self._started:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()

    # The pixel format must be changed when updating the sensor mode, so doing
    # both at the same time
    self._cam.video_configuration.sensor = {'output_size': mode.size,
                                            'bit_depth': mode.bit_depth}
    self._cam.video_configuration.main = {'format': pixel_format,
                                          'size': mode.size}

    # The possible size of the software ROI must be updated when changing the
    # sensor mode
    self.reload_software_roi(mode.size[0], mode.size[1])
    
    if self._started:
      # Re-applying the configuration to make the changes effective
      self._cam.configure('video')

      # Re-starting the image acquisition
      self.log(logging.DEBUG, "Starting image acquisition")
      self._cam.start_recording(self._encoder, self._output)
  
  def _set_pixel_format(self, pixel_format: str) -> None:
    """Sets the used pixel format to the desired value."""

    # Handles the case when the camera is initially started with an unsupported
    # pixel format, like XRGB8888
    if pixel_format not in PIXEL_FORMATS:
      pixel_format = 'RGB888'
    
    mode = self._current_sensor_mode

    # The pixel format cannot be adjusted while the camera is running
    if self._started:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()

    # The mode size is needed for setting the pixel format, so configuring the
    # sensor mode at the same time
    self._cam.video_configuration.sensor = {'output_size': mode.size,
                                            'bit_depth': mode.bit_depth}
    self._cam.video_configuration.main = {'format': pixel_format,
                                          'size': mode.size}

    if self._started:
      # Re-applying the configuration to make the changes effective
      self._cam.configure('video')

      # Re-starting the image acquisition
      self.log(logging.DEBUG, "Starting image acquisition")
      self._cam.start_recording(self._encoder, self._output)
  
  def _set_auto_exposure(self, auto_exposure: bool) -> None:
    """Enables or disables the auto exposure feature."""
    
    if not self._started:
      self._cam.video_configuration.controls['AeEnable'] = auto_exposure
    else:
      self._cam.set_controls({'AeEnable': auto_exposure})
  
  def _set_auto_focus(self, mode: str) -> None:
    """Sets the autofocus mode of the camera to the desired value."""
    
    mode_enum = AUTO_FOCUS_MODE_INV[mode]
    if not self._started:
      self._cam.video_configuration.controls['AfMode'] = mode_enum
    else:
      self._cam.set_controls({'AfMode': mode_enum})
  
  def _set_gain(self, gain: float) -> None:
    """Sets the gain of the image to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['AnalogueGain'] = gain
    else:
      self._cam.set_controls({'AnalogueGain': gain})
  
  def _set_auto_white_balance(self, auto_white_balance: bool) -> None:
    """Enables or disables the auto white balance feature."""
    
    if not self._started:
      self._cam.video_configuration.controls['AwbEnable'] = auto_white_balance
    else:
      self._cam.set_controls({'AwbEnable': auto_white_balance})
  
  def _set_brightness(self, brightness: float) -> None:
    """Sets the brightness of the image to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['Brightness'] = brightness
    else:
      self._cam.set_controls({'Brightness': brightness})
  
  def _set_contrast(self, contrast: float) -> None:
    """Sets the contrast of the image to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['Contrast'] = contrast
    else:
      self._cam.set_controls({'Contrast': contrast})
  
  def _set_exposure_time(self, exposure: float) -> None:
    """Sets the exposure time of the camera to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['ExposureTime'] = exposure
    else:
      self._cam.set_controls({'ExposureTime': exposure})
  
  def _set_auto_exposure_value(self, value: float) -> None:
    """Sets the auto-exposure target to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['ExposureValue'] = value
    else:
      self._cam.set_controls({'ExposureValue': value})
  
  def _set_lens_position(self, position: float) -> None:
    """Sets the position of the motorized lens to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['LensPosition'] = position
    else:
      self._cam.set_controls({'LensPosition': position})
  
  def _set_saturation(self, saturation: float) -> None:
    """Sets the saturation of the image to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['Saturation'] = saturation
    else:
      self._cam.set_controls({'Saturation': saturation})
  
  def _set_sharpness(self, sharpness: float) -> None:
    """Sets the sharpness of the image to the desired value."""
    
    if not self._started:
      self._cam.video_configuration.controls['Sharpness'] = sharpness
    else:
      self._cam.set_controls({'Sharpness': sharpness})
  
  def _get_sensor_mode(self) -> str:
    """Returns the name of the current sensor mode."""
    
    sensor_config = self._cam.video_configuration.sensor
    mode_str = (f"{sensor_config.output_size[0]}x"
                f"{sensor_config.output_size[1]}, "
                f"{sensor_config.bit_depth}bits")
    return self._get_mode_name_from_str(mode_str)
  
  def _get_pixel_format(self) -> Literal['YUV420', 'RGB888']:
    """Returns the name of the current pixel format."""
    
    return self._cam.video_configuration.main.format
    
  def _get_mode_name_from_str(self, mode_str: str) -> str:
    """Convenience method returning the full name of a sensor mode using the
    beginning of its name."""
    
    try:
      return [mode.name for mode in self._sensor_modes 
              if mode.name.startswith(mode_str)][0]
    except IndexError:
      raise RuntimeError(f"Unhandled sensor mode read from camera: {mode_str}")
  
  def _get_mode_from_str(self, mode_str: str) -> SensorMode:
    """Convenience method returning a SensorMode object using the beginning of
    its name."""
    
    try:
      return [mode for mode in self._sensor_modes 
              if mode.name.startswith(mode_str)][0]
    except IndexError:
      raise RuntimeError(f"Unhandled sensor mode read from camera: {mode_str}")

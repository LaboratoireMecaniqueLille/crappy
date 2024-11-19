# coding: utf-8

import numpy as np
from typing import Optional, Any, Union
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
except (ModuleNotFoundError, ImportError):
  controls = OptionalModule("libcamera")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

AUTO_FOCUS_MODE = {
    controls.AfModeEnum.Continuous: 'Continuous',
    controls.AfModeEnum.Manual: 'Manual',
    controls.AfModeEnum.Auto: 'Auto'}
AUTO_FOCUS_MODE_INV = {
    val: key for key, val in AUTO_FOCUS_MODE.items()}

PIXEL_FORMATS = ('YUV420', 'RGB888')


class CrappyEncoder(Encoder):
  """"""
  
  def outputframe(self, array: MappedArray, metadata: dict[str, Any]) -> None:
    """"""
    
    if hasattr(self, '_output_lock'):
      with self._output_lock:
        for out in self._output:
          out.outputframe(array, metadata)
    else:
      for out in self._output:
        out.outputframe(array, metadata)
  
  def _encode(self, stream, request: CompletedRequest) -> None:
    """"""
    
    if isinstance(stream, str):
      stream = request.stream_map[stream]
    metadata = request.get_metadata()
    with MappedArray(request, stream, reshape=True, write=True) as array:
      self.outputframe(array, metadata)


class CrappyOutput(Output):
  """"""
  
  def __init__(self, 
               out: dict[str, Union[Optional[np.ndarray], dict[str, Any]]],
               lock: RLock) -> None:
    """"""
    
    super().__init__(pts=None)
    self._shared: dict[str, Union[Optional[np.ndarray], dict[str, Any]]] = out
    self._frame_count: int = 0
    self._lock = lock
  
  def outputframe(self, array: MappedArray, metadata: dict[str, Any]) -> None:
    """"""
    
    to_retrieve = ('SensorTimestamp', 'ExposureTime', 
                   'AnalogueGain', 'DigitalGain')
    with self._lock:
      self._shared['array'] = array.array.copy()
      self._shared['metadata'] = {key: val for key, val in metadata.items() 
                                  if key in to_retrieve and key in metadata}
      t = time()
      self._shared['metadata'] |= {'ImageUniqueID': self._frame_count,
                                   't(s)': t,
                                   'DateTimeOriginal': 
                                   strftime("%Y:%m:%d %H:%M:%S", gmtime(t)),
                                   'SubsecTimeOriginal': f'{t % 1:.6f}'}
      self._frame_count += 1


@dataclass
class SensorMode:
  """"""
  
  format: str
  unpacked: str
  bit_depth: int
  size: tuple[int, int]
  fps: float
  crop_limits: tuple[int, int, int, int]
  exposure_limits: tuple[int, ...]
  
  @property
  def name(self) -> str:
    """"""
    
    return (f"{self.size[0]}x{self.size[1]}, "
            f"{self.bit_depth}bits, {self.fps:.0f}fps")
    

class RaspberryPiCamera2(Camera):
  """"""
  
  def __init__(self) -> None:
    """"""
    
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
    
    self._shared: dict[str, Union[Optional[np.ndarray], dict[str, Any]]] = {
        'array': None,
        'metadata': dict()}
    self._lock: RLock = RLock()
    self._last_id: int = -1
  
  def open(self,
           camera_num: int = 0,
           sensor_mode: Optional[str] = None,
           pixel_format: Optional[str] = None,
           grey_level_images: Optional[bool] = None,
           auto_exposure: Optional[bool] = None,
           auto_focus: Optional[str] = None,
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
    """"""
    
    # Opening the chosen camera
    self.log(logging.DEBUG, f"Opening the camera n°{camera_num}")
    self._cam = Picamera2(camera_num=camera_num)
    available_controls = self._cam.camera_controls
    
    self._sensor_modes = [SensorMode(mode['format'],
                                     mode['unpacked'],
                                     mode['bit_depth'],
                                     mode['size'],
                                     mode['fps'],
                                     mode['crop_limits'],
                                     mode['exposure_limits']) 
                          for mode in self._cam.sensor_modes]
    
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
    
    self.add_bool_setting('grey_level_images',
                          None,
                          None,
                          False)
    
    if 'AeEnable' in available_controls:
      self.add_bool_setting('auto_exposure', 
                            None,
                            self._set_auto_exposure, 
                            False)
    
    if 'AfMode' in available_controls:
      self.add_choice_setting('auto_focus',
                              AUTO_FOCUS_MODE.values(),
                              None,
                              self._set_auto_focus,
                              'Manual')
    
    if 'AnalogueGain' in available_controls:
      self.add_scale_setting('analog_gain', 
                             available_controls['AnalogueGain'][0], 
                             available_controls['AnalogueGain'][1], 
                             None, 
                             self._set_gain, 
                             available_controls['AnalogueGain'][0])
    
    if 'AwbEnable' in available_controls:
      self.add_bool_setting('auto_white_balance', 
                            None,
                            self._set_auto_white_balance, 
                            False)
    
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
    
    if 'LensPosition' in available_controls:
      self.add_scale_setting('lens_position', 
                             available_controls['LensPosition'][0], 
                             available_controls['LensPosition'][1],
                             None, 
                             self._set_lens_position, 
                             available_controls['LensPosition'][0])
    
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
    
    self.add_software_roi(self._current_sensor_mode.size[0],
                          self._current_sensor_mode.size[1])
    
    # Collecting the kwargs to set and setting them
    to_set = {name: arg for name, arg in zip(
        ('sensor_mode', 'pixel_format', 'grey_level_images', 'auto_exposure',
         'auto_focus', 'analog_gain', 'ROI_width', 'ROI_height', 'ROI_x',
         'ROI_y', 'auto_white_balance', 'brightness', 'contrast', 
         'exposure_time', 'auto_exposure_value', 'lens_position', 'saturation', 
         'sharpness'),
        (sensor_mode, pixel_format, grey_level_images, auto_exposure, auto_focus,
         analog_gain, soft_roi_width, soft_roi_height, soft_roi_x,
         soft_roi_y, auto_white_balance, brightness, contrast, exposure_time, 
         auto_exposure_value, lens_position, saturation, sharpness))
              if arg is not None and name in self.settings}
    
    self.set_all(**to_set)
    
    self._cam.configure('video')
    
    self._encoder = CrappyEncoder()
    self._output = CrappyOutput(out=self._shared, lock = self._lock)
    
    self.log(logging.DEBUG, "Starting image acquisition")
    self._cam.start_recording(self._encoder, self._output)
    self._started = True
    
    # arguments in open
  
  def get_image(self) -> Optional[tuple[dict[str, Any], np.ndarray]]:
    """"""
    
    with self._lock:
      if (self._shared['array'] is not None
          and self._shared['metadata'] 
          and self._shared['metadata']['ImageUniqueID'] > self._last_id):
        self._last_id = self._shared['metadata']['ImageUniqueID']
        
        if (self.settings['pixel_format']._value_no_getter == 'YUV420' 
            and len(self._shared['array'].shape) == 2):
          img = cv2.cvtColor(self._shared['array'], cv2.COLOR_YUV420p2RGB)
        else:
          img = self._shared['array']
        
        if self.settings['grey_level_images'].value:
          img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return self._shared['metadata'], img
  
  def close(self) -> None:
    """"""
    
    if self._cam is not None:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()
      self.log(logging.DEBUG, "Closing the connection to the camera")
      self._cam.close()
  
  def _set_sensor_mode(self, mode_str: str) -> None:
    """"""
    
    mode = [mode for mode in self._sensor_modes if mode.name == mode_str][0]
    self._current_sensor_mode = mode
    pixel_format = self._get_pixel_format()
    
    if self._started:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()
    
    self._cam.video_configuration.sensor = {'output_size': mode.size,
                                            'bit_depth': mode.bit_depth}
    self._cam.video_configuration.main = {'format': pixel_format,
                                          'size': mode.size}
    
    self.reload_software_roi(mode.size[0], mode.size[1])
    
    if self._started:
      self._cam.configure('video')
    
      self.log(logging.DEBUG, "Starting image acquisition")
      self._cam.start_recording(self._encoder, self._output)
  
  def _set_pixel_format(self, pixel_format: str) -> None:
    """"""
    
    if pixel_format not in PIXEL_FORMATS:
      pixel_format = 'RGB888'
    
    mode = self._current_sensor_mode
    
    if self._started:
      self.log(logging.DEBUG, "Stopping image acquisition")
      self._cam.stop_recording()
    
    self._cam.video_configuration.sensor = {'output_size': mode.size,
                                            'bit_depth': mode.bit_depth}
    self._cam.video_configuration.main = {'format': pixel_format,
                                          'size': mode.size}
    
    if self._started:
      self._cam.configure('video')
    
      self.log(logging.DEBUG, "Starting image acquisition")
      self._cam.start_recording(self._encoder, self._output)
  
  def _set_auto_exposure(self, auto_exposure: bool) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['AeEnable'] = auto_exposure
    else:
      self._cam.set_controls({'AeEnable': auto_exposure})
  
  def _set_auto_focus(self, mode: str) -> None:
    """"""
    
    mode_enum = AUTO_FOCUS_MODE_INV[mode]
    if not self._started:
      self._cam.video_configuration.controls['AfMode'] = mode_enum
    else:
      self._cam.set_controls({'AfMode': mode_enum})
  
  def _set_gain(self, gain: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['AnalogueGain'] = gain
    else:
      self._cam.set_controls({'AnalogueGain': gain})
  
  def _set_auto_white_balance(self, auto_white_balance: bool) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['AwbEnable'] = auto_white_balance
    else:
      self._cam.set_controls({'AwbEnable': auto_white_balance})
  
  def _set_brightness(self, brightness: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['Brightness'] = brightness
    else:
      self._cam.set_controls({'Brightness': brightness})
  
  def _set_contrast(self, contrast: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['Contrast'] = contrast
    else:
      self._cam.set_controls({'Contrast': contrast})
  
  def _set_exposure_time(self, exposure: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['ExposureTime'] = exposure
    else:
      self._cam.set_controls({'ExposureTime': exposure})
  
  def _set_auto_exposure_value(self, value: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['ExposureValue'] = value
    else:
      self._cam.set_controls({'ExposureValue': value})
  
  def _set_lens_position(self, position: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['LensPosition'] = position
    else:
      self._cam.set_controls({'LensPosition': position})
  
  def _set_saturation(self, saturation: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['Saturation'] = saturation
    else:
      self._cam.set_controls({'Saturation': saturation})
  
  def _set_sharpness(self, sharpness: float) -> None:
    """"""
    
    if not self._started:
      self._cam.video_configuration.controls['Sharpness'] = sharpness
    else:
      self._cam.set_controls({'Sharpness': sharpness})
  
  def _get_sensor_mode(self) -> str:
    """"""
    
    sensor_config = self._cam.video_configuration.sensor
    mode_str = (f"{sensor_config.output_size[0]}x"
                f"{sensor_config.output_size[1]}, "
                f"{sensor_config.bit_depth}bits")
    return self._get_mode_name_from_str(mode_str)
  
  def _get_pixel_format(self) -> str:
    """"""
    
    return self._cam.video_configuration.main.format
    
  def _get_mode_name_from_str(self, mode_str: str) -> str:
    """"""
    
    try:
      return [mode.name for mode in self._sensor_modes 
              if mode.name.startswith(mode_str)][0]
    except IndexError:
      raise RuntimeError(f"Unhandled sensor mode read from camera: {mode_str}")
  
  def _get_mode_from_str(self, mode_str: str) -> SensorMode:
    """"""
    
    try:
      return [mode for mode in self._sensor_modes 
              if mode.name.startswith(mode_str)][0]
    except IndexError:
      raise RuntimeError(f"Unhandled sensor mode read from camera: {mode_str}")

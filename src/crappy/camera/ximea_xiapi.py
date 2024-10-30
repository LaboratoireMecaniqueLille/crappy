# coding: utf-8

from time import time, strftime, gmtime
from typing import Optional, Any, Literal
import numpy as np
import logging
from warnings import warn

from .meta_camera import Camera
from .._global import OptionalModule

try:
  from ximea import xiapi
except (ModuleNotFoundError, ImportError):
  xiapi = OptionalModule("ximea", "To use XiAPI cameras, please install the "
                         "official ximea Python module")

# Camera models that were added in Crappy
SUPPORTED = ('MQ042MG-CM', 'MC124MG-SY-UB')

# Available data formats for the supported cameras
DATA_FORMATS = {'MQ042MG-CM': {'XI_MONO8': 'Mono (8 bits)',
                               'XI_MONO16': 'Mono (16 bits)',
                               'XI_RAW8': 'Raw (8 bits)',
                               'XI_RAW16': 'Raw (16 bits)'},
                'MC124MG-SY-UB': {'XI_MONO8': 'Mono (8 bits)',
                                  'XI_MONO16': 'Mono (16 bits)',
                                  'XI_RAW8': 'Raw (8 bits)',
                                  'XI_RAW16': 'Raw (16 bits)'}
                }
DATA_FORMATS_INV = {num: dict(zip(dic.values(), dic.keys()))
                    for num, dic in DATA_FORMATS.items()}

# Available framerate modes for the supported cameras
FRAMERATE_MODES = {'MQ042MG-CM': 
                   {'XI_ACQ_TIMING_MODE_FREE_RUN': 'Free run',
                    'XI_ACQ_TIMING_MODE_FRAME_RATE': 'Framerate target'},
                   'MC124MG-SY-UB':
                   {'XI_ACQ_TIMING_MODE_FREE_RUN': 'Free run',
                    'XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT': 'Framerate limit'}
                   }
FRAMERATE_MODES_INV = {num: dict(zip(dic.values(), dic.keys()))
                       for num, dic in FRAMERATE_MODES.items()}

# Available downsampling modes for the supported cameras
DOWNSAMPLING_MODES = {'MC124MG-SY-UB': {'XI_DWN_1x1': '1x1',
                                        'XI_DWN_2x2': '2x2'}}
DOWNSAMPLING_MODES_INV = {num: dict(zip(dic.values(), dic.keys()))
                          for num, dic in DOWNSAMPLING_MODES.items()}


class XiAPI(Camera):
  """This class can read images from any of the Ximea cameras.

  It heavily relies on the :mod:`ximea` module, distributed by Ximea, which is
  itself a wrapper around the XiAPI low-level library.

  Note:
    Both the Python module and the camera drivers have to be installed from
    Ximea's website in order for this class to run.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Xiapi* to *XiAPI*
  """

  def __init__(self) -> None:
    """Instantiates a Ximea Camera and a Ximea Image object."""

    super().__init__()

    self._cam = None
    self._started: bool = False
    self._model: Optional[str] = None
    self._timeout: Optional[int] = None

    self._cam = xiapi.Camera()
    self._img = xiapi.Image()

    # Stores the last requested or read trigger mode value
    self._trig = 'Free run'

  def open(self,
           serial_number: Optional[str] = None,
           timeout: Optional[int] = None,
           trigger: Literal['Free run', 'Hdw after config', 'Hardware'] = None,
           data_format: Literal['Mono (8 bits)', 'Mono (16 bits)',
                                'Raw (8 bits)', 'Raw (16 bits)'] = None,
           exposure_time_us: Optional[int] = None,
           gain: Optional[float] = None,
           auto_exposure_auto_gain: Optional[bool] = None,
           gamma_y: Optional[float] = None,
           gamma_c: Optional[float] = None,
           sharpness: Optional[float] = None,
           image_width: Optional[int] = None,
           image_height: Optional[int] = None,
           x_offset: Optional[int] = None,
           y_offset: Optional[int] = None,
           framerate_mode: Literal['Free run', 'Framerate target',
                                   'Framerate limit'] = None,
           framerate: Optional[float] = None,
           downsampling_mode: Literal['1x1', '2x2'] = None,
           width: Optional[int] = None,
           height: Optional[int] = None,
           xoffset: Optional[int] = None,
           yoffset: Optional[int] = None,
           exposure: Optional[float] = None,
           AEAG: Optional[bool] = None,
           external_trig: Optional[bool] = None) -> None:
    """Opens the connection to the camera, instantiates the available settings
    and starts the acquisition.

    Also sets custom values for the settings if provided by the user, otherwise
    sets them to their default.

    Args:
      serial_number: A :obj:`str` containing the serial number of the camera to
        open, in case several cameras are connected. If not provided and
        several cameras are available, one of them will be opened randomly.

        .. versionchanged:: 2.0.0 renamed from *sn* to *serial_number*

      timeout: The number of milliseconds the camera is allowed to wait for an
        image before raising a :exc:`TimeoutError`, as an :obj:`int`. Mostly
        useful when using an external trigger, or a very long exposure time.
        The default is 5000ms.

        .. versionadded:: 2.0.2

      trigger: Drives the trigger status of the camera. Can be either
        `'Free run'`, `'Hdw after config'`, or `'Hardware'`. The default is
        `'Free run'`.

        .. versionadded:: 2.0.0

      data_format: The data format to use for acquisition as a :obj:`str` (e.g.
        `'Mono (8 bits)'`). The available formats depend on the model of the
        camera. This setting is only implemented for a subset of all the Ximea
        cameras, get in touch with the maintainers to implement in on other
        models.

        .. versionadded:: 2.0.2

      exposure_time_us: The exposure time to use for acquiring images, in µs.
        Only has an effect if ``auto_exposure_auto_gain`` is set to
        :obj:`False`, and if ``framerate_mode`` is not in `'Free run'`.
        Depending on the camera might set the *target* or the *limit*
        framerate. The range of values depends on the model of the driven
        camera. This setting is only implemented for a subset of all the Ximea
        cameras, get in touch with the maintainers to implement in on other
        models.

        .. versionadded:: 2.0.2

      gain: The gain in dB to apply to the acquired images, as a :obj:`float`.
        Only has an effect if ``auto_exposure_auto_gain`` is set to
        :obj:`False`. The valid range of values depends on the model of the
        driven camera.
      auto_exposure_auto_gain: When set to :obj:`True`, overwrites the
        ``exposure_time_us`` and ``gain`` values and drives the exposure and
        the gain automatically.

        .. versionadded:: 2.0.2

      gamma_y: The luminosity gamma value, for color images only, as a
        :obj:`float`. The valid range of values depends on the model of the
        driven camera.

        .. versionadded:: 2.0.2

      gamma_c: The chromaticity gamma value, for color images only, as a
        :obj:`float`. The valid range of values depends on the model of the
        driven camera.

        .. versionadded:: 2.0.2

      sharpness: The sharpness strength value, for color images only, as a
        :obj:`float`. The valid range of values depends on the model of the
        driven camera.
        
        .. versionadded:: 2.0.2

      image_width: The width of the ROI to acquire in pixels, as an :obj:`int`.
        The valid range of values depends on the model of the driven camera, 
        and on the ``downsampling_mode`` if supported.
        
        .. versionadded:: 2.0.2
        
      image_height: The height of the ROI to acquire in pixels, as an 
        :obj:`int`. The valid range of values depends on the model of the 
        driven camera, and on the ``downsampling_mode`` if supported.
        
        .. versionadded:: 2.0.2
        
      x_offset: The X offset of the ROI to acquire in pixels, as an :obj:`int`. 
        The valid range of values depends on the model of the driven camera, 
        on the selected ``width``, and on the ``downsampling_mode`` if 
        supported.
        
        .. versionadded:: 2.0.2
        
      y_offset: The Y offset of the ROI to acquire in pixels, as an :obj:`int`. 
        The valid range of values depends on the model of the driven camera, 
        on the selected ``height``, and on the ``downsampling_mode`` if 
        supported.
        
        .. versionadded:: 2.0.2
        
      framerate_mode: The mode for driving the acquisition framerate. Can be 
        one of `'Free run'`, `'Framerate target'` or `'Framerate limit'`, 
        but not all camera models support the three options. This setting is 
        only implemented for a subset of all the Ximea cameras, get in touch 
        with the maintainers to implement in on other models.
        
        .. versionadded:: 2.0.2
        
      framerate: Either the target or the limit framerate for image 
        acquisition, depending on the ``framerate_mode`` value, as a 
        :obj:`float`. The valid range of values depends on the model of the 
        driven camera, and on the ``exposure_time_us`` value. Only has an 
        effect if ``framerate_mode`` is not set to `'Free run'`. This setting 
        is only implemented for a subset of all the Ximea cameras, get in touch 
        with the maintainers to implement in on other models.
        
        .. versionadded:: 2.0.2
        
      downsampling_mode: The downsampling mode to apply to the acquired images,
        as a :obj:`str` (e.g. `'2x2'`). This setting is only implemented for a 
        subset of all the Ximea cameras, get in touch with the maintainers to 
        implement in on other models.
        
        .. versionadded:: 2.0.2
        
      width: .. deprecated:: 2.0.2 Use ``image_width`` instead.
      height: .. deprecated:: 2.0.2 Use ``image_height`` instead.
      xoffset: .. deprecated:: 2.0.2 Use ``x_offset`` instead.
      yoffset: .. deprecated:: 2.0.2 Use ``y_offset`` instead.
      exposure: .. deprecated:: 2.0.2 Use ``exposure_time_us`` instead.
      AEAG: .. deprecated:: 2.0.2 Use ``auto_exposure_auto_gain`` instead.
      external_trig: .. deprecated:: 2.0.0 Use ``trigger`` instead.
    """

    # Managing the deprecated arguments
    if width is not None:
      warn("The width argument is deprecated for the XiAPI camera, use "
           "image_width instead !", FutureWarning)
      image_width = width
    if height is not None:
      warn("The height argument is deprecated for the XiAPI camera, use "
           "image_height instead !", FutureWarning)
      image_height = height
    if xoffset is not None:
      warn("The xoffset argument is deprecated for the XiAPI camera, use "
           "x_offset instead !", FutureWarning)
      x_offset = xoffset
    if yoffset is not None:
      warn("The yoffset argument is deprecated for the XiAPI camera, use "
           "y_offset instead !", FutureWarning)
      y_offset = yoffset
    if exposure is not None:
      warn("The exposure argument is deprecated for the XiAPI camera, use "
           "exposure_time_us instead !", FutureWarning)
      exposure_time_us = exposure
    if AEAG is not None:
      warn("The AEAG argument is deprecated for the XiAPI camera, use "
           "auto_exposure_auto_gain instead !", FutureWarning)
      auto_exposure_auto_gain = AEAG
    if external_trig is not None:
      warn("The external_trig argument is deprecated for the XiAPI camera, use"
           " trigger instead !", FutureWarning)
      trigger = 'Hdw after config'

    self._timeout = timeout

    # First, checking if there are connected cameras
    num_dev = self._cam.get_number_devices()
    if not num_dev:
      raise IOError("No connected Ximea camera was detected !")
    else:
      self.log(logging.INFO, f"Detected {num_dev} connected Ximea camera(s)")

    # Opening the camera by serial number if any was provided
    if serial_number is not None:
      self.log(logging.INFO, f"Opening the connection to the camera with "
                             f"serial number {serial_number}")
      self._cam.open_device_by_SN(serial_number)
    else:
      self.log(logging.INFO, "Opening the connection to a default camera")
      self._cam.open_device()

    # Checking the model of the opened camera and logging to the user
    self._model = self._cam.get_device_name().decode()
    self.log(logging.INFO, f"Opened the Ximea camera model "
                           f"{self._model}, with serial number "
                           f"{self._cam.get_device_sn().decode()}")
    if self._model not in SUPPORTED:
      self.log(logging.WARNING, f"The model {self._model} was never "
                                f"specifically implemented in Crappy, some if "
                                f"its features might not be available ! Get "
                                f"in touch with the maintainers to have it "
                                f"implemented.")

    # Data format parameter
    if self._model in DATA_FORMATS:
      self.add_choice_setting('data_format',
                              DATA_FORMATS[self._model].values(),
                              self._get_data_format,
                              self._set_data_format,
                              self._get_data_format())

    # Gain and exposure parameters
    self.add_scale_setting('exposure_time_us',
                           self._cam.get_exposure_minimum(),
                           min(self._cam.get_exposure_maximum(), 100000),
                           self._get_exp,
                           self._set_exp,
                           10000)
    self.add_scale_setting('gain',
                           self._cam.get_gain_minimum(),
                           self._cam.get_gain_maximum(),
                           self._get_gain,
                           self._set_gain,
                           0.0)
    self.add_bool_setting('auto_exposure_auto_gain',
                          self._get_aeag,
                          self._set_aeag,
                          False)

    # Color-related parameters
    self.add_scale_setting('gamma_y',
                           self._cam.get_gammaY_minimum(),
                           self._cam.get_gammaY_maximum(),
                           self._get_gamma_y,
                           self._set_gamma_y,
                           self._cam.get_gammaY(),
                           self._cam.get_gammaY_increment())
    self.add_scale_setting('gamma_c',
                           self._cam.get_gammaC_minimum(),
                           self._cam.get_gammaC_maximum(),
                           self._get_gamma_c,
                           self._set_gamma_c,
                           self._cam.get_gammaC(),
                           self._cam.get_gammaC_increment())
    self.add_scale_setting('sharpness',
                           self._cam.get_sharpness_minimum(),
                           self._cam.get_sharpness_maximum(),
                           self._get_sharpness,
                           self._set_sharpness,
                           self._cam.get_sharpness(),
                           self._cam.get_sharpness_increment())

    # ROI parameters
    self.add_scale_setting('image_width',
                           self._cam.get_width_minimum(),
                           self._cam.get_width_maximum(),
                           self._get_w,
                           self._set_w,
                           self._get_w(),
                           self._cam.get_width_increment())
    self.add_scale_setting('image_height',
                           self._cam.get_height_minimum(),
                           self._cam.get_height_maximum(),
                           self._get_h,
                           self._set_h,
                           self._get_h(),
                           self._cam.get_height_increment())
    self.add_scale_setting('x_offset',
                           self._cam.get_offsetX_minimum(),
                           self._cam.get_offsetX_maximum(),
                           self._get_ox,
                           self._set_ox,
                           self._get_ox(),
                           self._cam.get_offsetX_increment())
    self.add_scale_setting('y_offset',
                           self._cam.get_offsetY_minimum(),
                           self._cam.get_offsetY_maximum(),
                           self._get_oy,
                           self._set_oy,
                           self._get_oy(),
                           self._cam.get_offsetY_increment())

    # External trigger parameter
    self.add_trigger_setting(self._get_extt, self._set_ext_trig)

    # Framerate parameters
    if self._model in FRAMERATE_MODES:
      self.add_choice_setting('framerate_mode',
                              FRAMERATE_MODES[self._model].values(),
                              self._get_framerate_mode,
                              self._set_framerate_mode,
                              self._get_framerate_mode())
      self.add_scale_setting('framerate',
                             self._cam.get_framerate_minimum(),
                             min(self._cam.get_framerate_maximum(), 500),
                             self._get_framerate,
                             self._set_framerate,
                             self._cam.get_framerate(),
                             self._cam.get_framerate_increment())

    # Downsampling parameter
    if self._model in DOWNSAMPLING_MODES:
      self.add_choice_setting('downsampling_mode',
                              DOWNSAMPLING_MODES[self._model].values(),
                              self._get_downsampling_mode,
                              self._set_downsampling_mode,
                              self._get_downsampling_mode())

    # Collecting the kwargs to set and setting them
    to_set = {name: arg for name, arg in zip(
        ('exposure_time_us', 'gain', 'auto_exposure_auto_gain', 'gamma_y',
         'gamma_c', 'sharpness', 'image_width', 'image_height', 'x_offset',
         'y_offset', 'framerate_mode', 'framerate', 'downsampling_mode',
         'trigger'),
        (exposure_time_us, gain, auto_exposure_auto_gain, gamma_y, gamma_c,
         sharpness, image_width, image_height, x_offset, y_offset,
         framerate_mode, framerate, downsampling_mode, trigger))
              if arg is not None}

    if self._model in DATA_FORMATS and data_format is not None:
      to_set['data_format'] = data_format

    self.set_all(**to_set)

    # Starting the acquisition
    self.log(logging.INFO, "Starting the image acquisition")
    self._cam.start_acquisition()
    self._started = True

  def get_image(self) -> tuple[dict[str, Any], np.ndarray]:
    """Reads a frame from the camera, and returns it along with its metadata.

    The acquired metadata contains the following fields :

    * `'t(s)'`: The current timestamp as returned by :obj:`time.time`.
    * `'DateTimeOriginal'`: The current date up to the second as a valid exif
      tag, based on the value of `'t(s)'`.
    * `'SubsecTimeOriginal'`: The sub-second part of the current date as a
      valid exif tag, based on the value of `'t(s)'`.
    * `'XimeaSec'`: The number of seconds the camera has been up.
    * `'XimeaUSec'`: The decimal part of the above field, value in µs.
    * `'ImageWidth'`: The width of the acquired image, in pixels.
    * `'ImageHeight'`: The height of the acquired image, in pixels.
    * `'ExposureTime'`: The exposure time of the acquired image, in µs.
    * `'AbsoluteOffsetX'`: The offset of the acquired ROI along the X axis, in
      pixels.
    * `'AbsoluteOffsetY'`: The offset of the acquired ROI along the Y axis, in
      pixels.
    * `'DownsamplingX'`: The downsampling factor along the X axis.
    * `'DownsamplingY'`: The downsampling factor along the Y axis.
    * `'ImageUniqueID'`: The index of the acquired image, as returned by the
      camera.

    """

    if self._timeout is not None:
      self._cam.get_image(self._img, timeout=self._timeout)
    else:
      self._cam.get_image(self._img)

    # Returning metadata from the camera long with the captured image
    t = time()
    metadata = {'t(s)': t,
                'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S", gmtime(t)),
                'SubsecTimeOriginal': f'{t % 1:.6f}',
                'XimeaSec': self._img.tsSec,
                'XimeaUSec': self._img.tsUSec,
                'ImageWidth': self._img.width,
                'ImageHeight': self._img.height,
                'ExposureTime': self._img.exposure_time_us,
                'AbsoluteOffsetX': self._img.AbsoluteOffsetX,
                'AbsoluteOffsetY': self._img.AbsoluteOffsetY,
                'DownsamplingX': self._img.DownsamplingX,
                'DownsamplingY': self._img.DownsamplingY,
                'ImageUniqueID': self._img.acq_nframe}

    return metadata, self._img.get_image_data_numpy()

  def close(self) -> None:
    """Closes the connection to the camera and releases the resources."""

    if self._cam is not None:
      self.log(logging.INFO, "Closing the connection to the camera")
      self._cam.close_device()

  def _get_data_format(self) -> Literal['Mono (8 bits)', 'Mono (16 bits)',
                                        'Raw (8 bits)', 'Raw (16 bits)']:
    """Returns the current data format of the acquired images."""

    return DATA_FORMATS[self._model][self._cam.get_imgdataformat()]

  def _get_exp(self) -> float:
    """Returns the exposure time, in microseconds."""

    return self._cam.get_exposure()

  def _get_aeag(self) -> bool:
    """Return the status of the Auto Exposure / Auto Gain setting."""

    return self._cam.is_aeag()

  def _get_gain(self) -> float:
    """Returns the gain, in dB."""

    return self._cam.get_gain()

  def _get_gamma_y(self) -> float:
    """Returns the current Gamma Y value of the camera."""

    return self._cam.get_gammaY()

  def _get_gamma_c(self) -> float:
    """Returns the current Gamma C value of the camera."""

    return self._cam.get_gammaC()
  
  def _get_sharpness(self) -> float:
    """Returns the current sharpness value of the camera."""

    return self._cam.get_sharpness()

  def _get_w(self) -> int:
    """Returns the width in pixels for selecting a region of interest."""

    return self._cam.get_width()

  def _get_h(self) -> int:
    """Returns the height in pixels for selecting a region of interest."""

    return self._cam.get_height()

  def _get_ox(self) -> int:
    """Returns the `x` offset in pixels for selecting a region of interest."""

    return self._cam.get_offsetX()

  def _get_oy(self) -> int:
    """Returns the `y` offset in pixels for selecting a region of interest."""

    return self._cam.get_offsetY()

  def _get_extt(self) -> Literal['Free run', 'Hdw after config', 'Hardware']:
    """Returns the current trigger mode value, and updates the last read
    trigger mode value if needed.

    The possible values for the trigger mode are `'Hardware'`, `'Free run'`,
    and `'Hdw after config'`.
    """

    if ((r := self._cam.get_trigger_source()) == 'XI_TRG_OFF' and
        self._trig == 'Hardware'):
      self._trig = 'Free run'
    elif r != 'XI_TRG_OFF' and self._trig != 'Hardware':
      self._trig = 'Hardware'
    return self._trig

  def _get_framerate(self) -> float:
    """Returns the current framerate value of the camera."""

    return self._cam.get_framerate()

  def _get_framerate_mode(self) -> Literal['Free run', 'Framerate target',
                                           'Framerate limit']:
    """Returns the current frame rate mode for the camera."""

    return FRAMERATE_MODES[self._model][self._cam.get_acq_timing_mode()]

  def _get_downsampling_mode(self) -> Literal['1x1', '2x2']:
    """Returns the current downsampling mode."""

    return DOWNSAMPLING_MODES[self._model][self._cam.get_downsampling()]

  def _set_data_format(self, fmt: Literal['Mono (8 bits)',
                                          'Mono (16 bits)',
                                          'Raw (8 bits)',
                                          'Raw (16 bits)']) -> None:
    """sets the requested data format."""

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    self._cam.set_imgdataformat(DATA_FORMATS_INV[self._model][fmt])

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

  def _set_exp(self, exposure: float) -> None:
    """Sets the exposure time, in microseconds."""

    if self._get_aeag():
      self.log(logging.WARNING, "Setting the exposure won't work as long as "
                                "the AEAG is enabled !")
    self._cam.set_exposure(exposure)

  def _set_aeag(self, val: bool) -> None:
    """Enables or disables the Auto Exposure / Auto Gain setting."""

    if val:
      self._cam.enable_aeag()
    else:
      self._cam.disable_aeag()

  def _set_gain(self, gain: float) -> None:
    """Sets the gain, in dB."""

    if self._get_aeag():
      self.log(logging.WARNING, "Setting the gain won't work as long as the "
                                "AEAG is enabled !")
    self._cam.set_gain(gain)

  def _set_gamma_y(self, gamma: float) -> None:
    """Sets the Gamma Y value on the camera."""

    self._cam.set_gammaY(gamma)

  def _set_gamma_c(self, gamma: float) -> None:
    """Sets the Gamma C value on the camera."""

    self._cam.set_gammaC(gamma)
  
  def _set_sharpness(self, sharpness: float) -> None:
    """Sets the sharpness value on the camera."""

    self._cam.set_sharpness(sharpness)

  def _set_w(self, width: int) -> None:
    """Sets the width in pixels for selecting a region of interest."""

    # Lowering the X offset if it conflicts with the new image width
    if self.x_offset + width > self.settings['image_width'].highest:
      self.x_offset = self.settings['image_width'].highest - width

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    # Setting the requested width and reloading the X offset
    self._cam.set_width(width)

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

    self.settings['x_offset'].reload(self._cam.get_offsetX_minimum(),
                                     self._cam.get_offsetX_maximum(),
                                     self._get_ox(),
                                     self._get_ox(),
                                     self._cam.get_offsetX_increment())

  def _set_h(self, height: int) -> None:
    """Sets the height in pixels for selecting a region of interest."""

    # Lowering the Y offset if it conflicts with the new image height
    if self.y_offset + height > self.settings['image_height'].highest:
      self.y_offset = self.settings['image_height'].highest - height

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    # Setting the requested height and reloading the Y offset
    self._cam.set_height(height)

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

    self.settings['y_offset'].reload(self._cam.get_offsetY_minimum(),
                                     self._cam.get_offsetY_maximum(),
                                     self._get_oy(),
                                     self._get_oy(),
                                     self._cam.get_offsetY_increment())

  def _set_ox(self, x_offset: int) -> None:
    """Sets the `x` offset in pixels for selecting a region of interest."""

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    self._cam.set_offsetX(x_offset)

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

  def _set_oy(self, y_offset: int) -> None:
    """Sets the `y` offset in pixels for selecting a region of interest."""

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    self._cam.set_offsetY(y_offset)

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

  def _set_ext_trig(self, trig: Literal['Free run', 'Hdw after config',
                                        'Hardware']) -> None:
    """Sets the requested trigger mode value, and updates the last requested
    trigger mode value.

    The possible values for the trigger mode are `'Hardware'`, `'Free run'`,
    and `'Hdw after config'`.
    """

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    if trig == 'Hardware':
      self._cam.set_gpi_mode('XI_GPI_TRIGGER')
      self._cam.set_trigger_source('XI_TRG_EDGE_RISING')
    else:
      self._cam.set_gpi_mode('XI_GPI_OFF')
      self._cam.set_trigger_source('XI_TRG_OFF')
      
    self._trig = trig
    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

  def _set_framerate_mode(self, mode: Literal['Free run', 'Framerate target',
                                              'Framerate limit']) -> None:
    """Sets the framerate mode for the camera to use."""

    self._cam.set_acq_timing_mode(FRAMERATE_MODES_INV[self._model][mode])

    self.settings['framerate'].reload(self._cam.get_framerate_minimum(),
                                      min(self._cam.get_framerate_maximum(),
                                          500),
                                      self._cam.get_framerate(),
                                      self._cam.get_framerate(),
                                      self._cam.get_framerate_increment())

  def _set_framerate(self, framerate) -> None:
    """Sets the target framerate value of the camera."""

    if self._get_framerate_mode() == 'Free run':
      self.log(logging.WARNING, "Setting the framerate won't work as long as "
                                "the camera is in free run mode !")
    self._cam.set_framerate(framerate)

  def _set_downsampling_mode(self, mode: Literal['1x1', '2x2']) -> None:
    """Sets the downsampling mode on the camera."""

    if self._started:
      self.log(logging.DEBUG, "Stopping the image acquisition")
      self._cam.stop_acquisition()

    self._cam.set_downsampling(DOWNSAMPLING_MODES_INV[self._model][mode])

    if self._started:
      self.log(logging.DEBUG, "Starting the image acquisition")
      self._cam.start_acquisition()

    self.settings['image_width'].reload(self._cam.get_width_minimum(),
                                        self._cam.get_width_maximum(),
                                        self._get_w(),
                                        self._get_w(),
                                        self._cam.get_width_increment())

    self.settings['image_height'].reload(self._cam.get_height_minimum(),
                                         self._cam.get_height_maximum(),
                                         self._get_h(),
                                         self._get_h(),
                                         self._cam.get_height_increment())

    self.settings['x_offset'].reload(self._cam.get_offsetX_minimum(),
                                     self._cam.get_offsetX_maximum(),
                                     self._get_ox(),
                                     self._get_ox(),
                                     self._cam.get_offsetX_increment())

    self.settings['y_offset'].reload(self._cam.get_offsetY_minimum(),
                                     self._cam.get_offsetY_maximum(),
                                     self._get_oy(),
                                     self._get_oy(),
                                     self._cam.get_offsetY_increment())

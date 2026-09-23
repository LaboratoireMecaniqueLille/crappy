# coding: utf-8

from collections.abc import Callable
import numpy as np
from collections import defaultdict
import logging
from time import time, strftime, gmtime
from types import MethodType
from typing import Any

from .block import VisionBlock
from ..camera import (deprecated_cameras, camera_dict, DummyCam,
                      moved_to_collection)
from ...tool.camera_config import CameraConfig
from ...camera import Camera as BaseCam
from ..._collection import (CollectionEntry, collection_registry,
                            load_collection_class)
from ..._global import CameraConfigError, PrepareError


class CameraSource(VisionBlock):
  """Acquires images from one Camera and publishes them to
  :class:`~crappy.blocks.vision.block.VisionBlock`.

  This Block drives one :class:`~crappy.camera.meta_camera.camera.Camera` and
  sends each acquired frame and its metadata through one or more output
  :class:`~crappy.links.img_link.ImageLink` objects. It accepts no input
  ImageLink and requires at least one output ImageLink. Acquisition is
  intentionally separated from processing, display, and recording. Connect
  processors, :class:`~crappy.blocks.vision.ImageDisplayer`, or
  :class:`~crappy.blocks.vision.ImageRecorder` to build the desired pipeline.

  For every published image, the Block also sends a dictionary through its
  regular output :class:`~crappy.links.link.Link` objects. The ``'t(s)'`` entry
  is the acquisition time relative to the beginning of the test,
  ``'img_index'`` is the Camera-provided ``'ImageUniqueID'``, and ``'meta'`` is
  the complete metadata dictionary. Regular input Links can provide a software
  trigger. When ``image_generator`` is used, they can additionally update its
  synthetic ``'Exx(%)'`` and ``'Eyy(%)'`` strain inputs.

  Before acquisition starts, the Block can open a generic
  :class:`~crappy.tool.camera_config.CameraConfig` window for previewing images
  and adjusting the Camera settings. Downstream processors may instead request
  specialized configuration windows, such as
  :class:`~crappy.tool.camera_config.DICVEConfig`. With ``config`` and
  ``allow_downstream_config`` enabled, these requests are run sequentially and
  their results are returned to the requesting Blocks. A required configuration
  request causes preparation to fail if interactive configuration is disabled.

  As an alternative to a physical Camera, ``image_generator`` can produce
  synthetic images from horizontal and vertical strain values. This mode is
  primarily intended for examples and development.

  Unlike :class:`~crappy.blocks.Camera`, this Block only performs acquisition.
  The older Camera Block combines acquisition with optional child processes
  for image processing, display, and recording.

  .. versionadded:: 2.1.0
  """

  cam_count: dict[str, int] = defaultdict(lambda: 0)

  def __init__(self,
               camera: str,
               transform: Callable[[np.ndarray], np.ndarray] | None = None,
               config: bool = True,
               allow_downstream_config: bool = True,
               image_generator: Callable[[float, float],
                                         np.ndarray] | None = None,
               software_trig_label: str | None = None,
               img_shape: tuple[int, int] | tuple[int, int, int] | None = None,
               img_dtype: str | None = None,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 100,
               **kwargs) -> None:
    """Sets the Camera, configuration, and acquisition options.

    Args:
      camera: Name of the :class:`~crappy.camera.meta_camera.camera.Camera` to
        use. Additional Camera-specific arguments can be supplied through
        ``kwargs``. This argument is ignored when ``image_generator`` is
        provided, and may then be an empty string.
      transform: Callable receiving each acquired image and returning the image
        to publish. It runs synchronously immediately after acquisition, so a
        costly transform can reduce acquisition frequency. Only the transformed
        image is sent, and its shape and dtype must match the prepared output
        buffer.
      config: If :obj:`True`, displays a
        :class:`~crappy.tool.camera_config.CameraConfig` window before the test
        when no specialized downstream request is present. The user can preview
        images and adjust the available
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
        values. Configuration also determines the output image shape and dtype.
        If :obj:`False`, both ``img_shape`` and ``img_dtype`` must be supplied.
      allow_downstream_config: Whether downstream VisionBlocks may replace the
        generic window with specialized configuration requests. Required
        requests can only be served when this argument and ``config`` are both
        :obj:`True`. Optional requests are declined when either is disabled.
      image_generator: Callable taking horizontal and vertical strain values in
        percent and returning a :class:`numpy.ndarray`. When provided, a dummy
        Camera exposes ``'Exx'`` and ``'Eyy'`` settings, the ``camera``
        argument is ignored, and incoming ``'Exx(%)'`` and ``'Eyy(%)'`` values
        update the generated image. This mode is primarily intended for
        examples and development.
      software_trig_label: Name of a label used as a software acquisition
        trigger. An image is acquired only after data containing this label
        arrives on a regular input Link, the value itself is ignored. This is
        not a precision trigger and should generally be kept below 10 Hz, use
        hardware triggering at higher rates if possible.
      img_shape: Shape of the published images. It is mandatory when ``config``
        is :obj:`False`. When configuration supplies a different shape, the
        configured value takes precedence.
      img_dtype: Dtype of the published images, as a string accepted by
        :class:`numpy.dtype`. It is mandatory when ``config`` is :obj:`False`.
        When configuration supplies a different dtype, the configured value
        takes precedence.
      display_freq: If :obj:`True`, periodically reports the achieved image
        acquisition frequency.
      debug: If :obj:`True`, displays all log messages, including
        :obj:`~logging.DEBUG` messages. If :obj:`False`, only displays messages
        at :obj:`~logging.INFO` level or higher. If :obj:`None`, disables
        logging for this Block.
      freq: Target acquisition-loop frequency. If :obj:`None`, loops as fast as
        possible. The Camera and processing time may limit the actual rate.
      **kwargs: Additional arguments forwarded to the selected Camera's
        :meth:`~crappy.camera.meta_camera.camera.Camera.open` method.
    """

    super().__init__(img_shape=img_shape,
                     img_dtype=img_dtype,
                     display_freq=display_freq,
                     debug=debug,
                     freq=freq)

    self._camera: BaseCam | None = None

    if not isinstance(camera, str):
      raise TypeError("camera must be a string")
    if not camera and image_generator is None:
      raise ValueError("camera must be a non-empty string")

    # Checking for deprecated names
    if camera in deprecated_cameras:
      raise NotImplementedError(
          f"The {camera} Camera was deprecated in version 2.0.0, and renamed "
          f"to {deprecated_cameras[camera]} ! Please update your code "
          f"accordingly and check the documentation for more information")

    # None means that this is an ordinary core or user-defined Camera
    self._collection_entry: CollectionEntry | None = None

    # Checking if the requested Camera exists in Crappy
    if image_generator is None:
      # Check if the requested Camera is part of crappy.collection
      entry = collection_registry.get("Camera", camera)
      # Cannot find the Camera in the list of available ones
      if camera not in camera_dict:
        # First option, the Camera should be loaded from crappy.collection
        if entry is not None:
          # This call raises early if the module cannot be loaded
          load_collection_class(entry, camera_dict)
          self._collection_entry = entry
        # Second case, the Camera was moved to crappy.collection but this
        # module was not imported
        elif camera in moved_to_collection:
          raise NotImplementedError(f"The Camera {camera} was moved to "
                                    f"crappy.collection. To use it, simply "
                                    f"add import crappy.collection at the "
                                    f"beginning of your script")
        # The name of the Camera simply cannot be found anywhere
        else:
          possible = ', '.join(sorted(camera_dict.keys()))
          raise ValueError(f"Unknown Camera name : {camera}! "
                           f"The currently available ones are: {possible}")
      # Case when the Camera was already loaded in a separate Block
      elif (entry is not None
            and camera_dict[camera].__module__ == entry.module):
        self._collection_entry = entry

      self._camera_name: str = camera
    else:
      self._camera_name: str = 'Image Generator'

    # Incrementing the count of opened cameras for this camera type
    CameraSource.cam_count[self._camera_name] += 1

    # Checking the validity of the provided arguments
    if transform is not None and not callable(transform):
      raise TypeError("When provided, transform must be a callable")
    if not isinstance(config, bool):
      raise TypeError("config must be a boolean")
    if not isinstance(allow_downstream_config, bool):
      raise TypeError("allow_downstream_config must be a boolean")
    if (software_trig_label is not None and
        (not isinstance(software_trig_label, str) or not software_trig_label)):
      raise ValueError("When provided, software_trig_label must be a "
                       "non-empty string")
    if image_generator is not None and not callable(image_generator):
      raise TypeError("When provided, image_generator must be a callable")

    # Setting the other attributes
    self._trig_label: str | None = software_trig_label
    self._config_cam: bool = config
    self._allow_downstream_config: bool = allow_downstream_config
    self._transform: Callable[[np.ndarray], np.ndarray] | None = transform
    self._image_generator: Callable[[float, float],
                                    np.ndarray] | None = image_generator
    self._camera_kwargs = kwargs

  def prepare(self) -> None:
    """Opens the Camera, runs configuration, and creates image buffers.

    This Block must have at least one output ImageLink and no input ImageLink.
    A physical Camera is instantiated and opened with the Camera-specific
    keyword arguments, or a dummy Camera is prepared around
    ``image_generator``. If interactive configuration is enabled, specialized
    downstream requests are handled in order. Otherwise, the generic Camera
    configuration window is used. Optional requests that cannot be handled are
    answered with :obj:`None`, while required requests abort preparation.

    The method also resolves ``'Hdw after config'`` trigger mode, verifies that
    the final output image shape and dtype are known, and delegates shared
    buffer creation to :class:`~crappy.blocks.vision.block.VisionBlock`.

    Raises:
      IOError: If the Block has an input ImageLink or no output ImageLink.
      RuntimeError: If the Camera is unavailable or a required configuration
        request cannot be handled.
      ValueError: If preparation synchronization objects or the final image
        format are unavailable.
      PrepareError: If another Block fails while configurations are running.
      CameraConfigError: If an interactive configuration window fails.
    """

    # Ensuring Link consistency
    if self.img_inputs:
      raise IOError("This VisionBlock does not support input ImageLink")
    if not self.img_outputs:
      raise IOError("This VisionBlock is useless without at least one output "
                    "ImageLink")

    # Case when the images are artificially generated and not acquired
    if self._image_generator is not None:
      self.log(logging.INFO, "Setting the image generator camera")
      self._camera = DummyCam()
      if self._camera is None:
        raise RuntimeError("The Camera wasn't set whereas it should be")
      self._camera.add_scale_setting('Exx', -100., 100., None, None, 0.)
      self._camera.add_scale_setting('Eyy', -100., 100., None, None, 0.)
      img = self._image_generator(0, 0)
      self._camera.add_software_roi(img.shape[1], img.shape[0])
      self._camera.set_all()

      def get_image(self_) -> tuple[float, np.ndarray]:
        """Returns a timestamp and an image from ``image_generator``."""

        return time(), self_.apply_soft_roi(self._image_generator(self_.Exx,
                                                                  self_.Eyy))

      self._camera.get_image = MethodType(get_image, self._camera)

    # Instantiating the Camera object for acquiring the images
    else:
      # Under the spawn multiprocessing start method, it is necessary to
      # re-load the modules from crappy.collection
      if self._collection_entry is not None:
        load_collection_class(self._collection_entry, camera_dict)

      self._camera = camera_dict[self._camera_name]()
      self.log(logging.INFO, f"Opening the {self._camera_name} Camera")
      if self._camera is None:
        raise RuntimeError("The Camera wasn't set whereas it should be")
      self._camera.open(**self._camera_kwargs)
      self.log(logging.INFO, f"Opened the {self._camera_name} Camera")

    if (self.config_requests_in and
        (not self._config_cam or not self._allow_downstream_config)):
      unhandled = [request.requester for request in self.config_requests_in
                   if request.required]
      if unhandled:
        raise RuntimeError(f"The combination of the config and "
                           f"allow_downstream_config arguments is leading to "
                           f"unhandled required config requests from Blocks "
                           f"{', '.join(unhandled)}, aborting early!\nTry to "
                           f"set config=True and allow_downstream_config=True,"
                           f" or provide configuration information to the "
                           f"downstream Blocks")

    # Displaying the configuration windows if required
    if self._config_cam:

      # Fail early in case of inconsistent state
      if self._ready_barrier is None:
        raise ValueError("The ready Barrier should be set at this point")
      if self._stop_event is None:
        raise ValueError("The stop Event should be initialized at this point")

      # Downstream config allowed and config requests received
      if self._allow_downstream_config:
        if self.config_requests_in:
          self.log(logging.INFO, "Running the configuration requests from "
                                 "downstream Blocks")
          for request in self.config_requests_in:

            # First check if we should proceed with the configurations
            if self._ready_barrier.broken or self._stop_event.is_set():
              raise PrepareError("An exception occurred in another Block, "
                                 "aborting")

            self.log(logging.DEBUG, f"Starting configure request from Block "
                                    f"{request.requester}")
            config = self.configure(self._camera, request.configurator,
                                    *request.args, **request.kwargs)
            self.log(logging.DEBUG, f"Got configuration result {config} for "
                                    f"Block {request.requester}")
            self.log(logging.DEBUG, f"Sending configuration result to Block "
                                    f"{request.requester}")
            self.send_config(request, config)

        # Downstream config allowed and no config requests received
        else:
          self.log(logging.INFO, "No config request from downstream Blocks, "
                                 "falling back to default config")
          self.default_configuration()

      # Downstream config not allowed
      if not self._allow_downstream_config:
        self.log(logging.INFO, "Performing default configuration since "
                               "allow_downstream_config is disabled")
        self.default_configuration()
        if self.config_requests_in:
          self.log(logging.INFO, "Declining incoming config requests since "
                                 "allow_downstream_config is disabled")
          for request in self.config_requests_in:
            if request.required:
              raise RuntimeError(f"Can't handle required configuration "
                                 f"request from Block {request.requester} as "
                                 f"allow_downstream_config is False, aborting")
            else:
              self.send_config(request, None)

    else:
      self.log(logging.INFO, "Skipping interactive configuration as requested")
      if self.config_requests_in:
        self.log(logging.INFO, "Declining incoming config requests since "
                               "config is disabled")
        for request in self.config_requests_in:
          if request.required:
            raise RuntimeError(f"Can't handle required configuration "
                               f"request from Block {request.requester} as "
                               f"config is False, aborting")
          else:
            self.send_config(request, None)

    # Setting the camera to 'Hardware' trig if it's in 'Hdw after config' mode
    if (self._camera.trigger_name in self._camera.settings and
        getattr(self._camera,
                self._camera.trigger_name) == 'Hdw after config'):
      self.log(logging.INFO, "Setting the trigger mode to Hardware")
      setattr(self._camera, self._camera.trigger_name, 'Hardware')

    # Ensuring a dtype and a shape were given for the image
    if self._img_dtype is None or self._img_shape is None:
      raise ValueError(f"Cannot launch the CameraSource for camera "
                       f"{self._camera_name} as the image shape and/or dtype "
                       f"wasn't specified.\n Please specify it in the args, or"
                       f" enable the configuration window.")

    # Mandatory otherwise the Block won't run
    super().prepare()

  def loop(self) -> None:
    """Acquires and publishes one image when the source is ready.

    Incoming regular-Link data first gates the optional software trigger and,
    in image-generator mode, updates the synthetic strain settings. The method
    then calls :meth:`~crappy.camera.meta_camera.camera.Camera.get_image`. If
    the Camera supplies only a timestamp, standard ``'DateTimeOriginal'``,
    ``'SubsecTimeOriginal'``, and ``'ImageUniqueID'`` metadata are generated.
    The ``'t(s)'`` timestamp is made relative to the test start before the
    optional image transform runs.

    The resulting image and metadata are published through all output
    ImageLinks. A dictionary containing ``'t(s)'``, ``'img_index'``, and the
    complete ``'meta'`` dictionary is also sent through regular output Links.
    If the trigger is absent or the Camera returns no image, the loop returns
    without publishing anything.

    Raises:
      RuntimeError: If the Camera has not been initialized.
      ValueError: If Camera metadata is not a dictionary containing ``'t(s)'``.
    """

    # Receiving the data from upstream Blocks
    data = self.recv_last_data(fill_missing=False)

    # Waiting for the trig label if one was given
    if self._trig_label is not None and self._trig_label not in data:
      # If requested, displays the FPS of the image acquisition
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    elif self._trig_label is not None and self._trig_label in data:
      self.log(logging.DEBUG, "Software trigger signal received")

    # Updating the image generator if one was provided
    if self._image_generator is not None:
      if 'Exx(%)' in data:
        self.log(logging.DEBUG, f"Setting Exx to {data['Exx(%)']}")
        self._camera.Exx = data['Exx(%)']
      if 'Eyy(%)' in data:
        self.log(logging.DEBUG, f"Setting Eyy to {data['Eyy(%)']}")
        self._camera.Eyy = data['Eyy(%)']

    # Grabbing the frame from the Camera object
    if self._camera is None:
      raise RuntimeError("The Camera wasn't set whereas it should be")
    if (ret := self._camera.get_image()) is None:
      self.log(logging.DEBUG, "No image grabbed in this loop")
      # If requested, displays the FPS of the image acquisition
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    self.log(logging.DEBUG, "Acquired an image during this loop")
    metadata, img = ret

    # Building the metadata dict if it was not provided
    if isinstance(metadata, float):
      metadata = {'t(s)': metadata,
                  'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S",
                                               gmtime(metadata)),
                  'SubsecTimeOriginal': f'{metadata % 1:.6f}',
                  'ImageUniqueID': self._sent_img_counter}

    # Making the timestamp relative to the beginning of the test
    if isinstance(metadata, dict) and 't(s)' in metadata:
      metadata['t(s)'] -= self.t0
    else:
      raise ValueError("At that point, the metadata must be a dictionary "
                       "containing a 't(s)' key")

    # Applying the transform function if one as provided
    if self._transform is not None:
      img = self._transform(img)

    # Sending the image to downstream Blocks
    self.send_img(metadata, img)

    # Sending information on the image through regular Links
    self.send({'t(s)': metadata['t(s)'],
               'img_index': metadata['ImageUniqueID'],
               'meta': metadata})

    # If requested, displays the FPS of the image acquisition
    if self.display_freq:
      self._print_freq(img_handled=True)

  def finish(self) -> None:
    """Closes the physical Camera and releases shared image resources.

    The Camera's :meth:`~crappy.camera.meta_camera.camera.Camera.close` method
    is skipped in image-generator mode. Shared-memory cleanup is then delegated
    to :class:`~crappy.blocks.vision.block.VisionBlock`.
    """

    # Closing the Camera object
    if self._image_generator is None and self._camera is not None:
      self.log(logging.INFO, f"Closing the {self._camera_name} Camera")
      self._camera.close()
      self.log(logging.INFO, f"Closed the {self._camera_name} Camera")

    # Mandatory for proper termination
    super().finish()

  def configure(self,
                camera: BaseCam,
                config_class: type[CameraConfig],
                *args,
                **kwargs) -> tuple[Any, ...] | None:
    """Runs one interactive configuration window for a Camera.

    The requested configurator receives this Block's logging settings, target
    frequency, image transform, and any request-specific arguments. Its final
    image shape and dtype replace the current output format, with a warning if
    either differs from an already known value.

    Args:
      camera: Open Camera instance to configure.
      config_class: :class:`~crappy.tool.camera_config.CameraConfig` subclass
        implementing the requested configuration window.
      *args: Positional arguments forwarded to *config_class*.
      **kwargs: Keyword arguments forwarded to *config_class*.

    Returns:
      The configuration data returned by
      :meth:`~crappy.tool.camera_config.CameraConfig.get_config`, or
      :obj:`None` if configuration is canceled.

    Raises:
      TypeError: If *camera* is not a Camera or *config_class* is not a
        :class:`~crappy.tool.camera_config.CameraConfig` subclass.
      RuntimeError: If the logging queue has not been initialized.
      CameraConfigError: If the configuration window fails.
      KeyboardInterrupt: If configuration is interrupted by the user.
    """

    # Preliminary general checks
    if not isinstance(camera, BaseCam):
      raise TypeError("camera must be an instance of Camera")
    if not issubclass(config_class, CameraConfig):
      raise TypeError("config_class must be a subclass of CameraConfig")

    config = None

    # Instantiating and starting the configuration window
    try:
      if self._log_queue is None:
        raise RuntimeError("Cannot start the configuration window because the "
                           "log_queue wasn't defined")
      self.log(logging.DEBUG, f"Starting configuration window of class "
                              f"{config_class} with camera {camera}, args "
                              f"{args}, kwargs {kwargs}")
      config = config_class(camera, self._log_queue, self._log_level,
                            self.freq, self._transform, *args, **kwargs)
      config.start()
      config.wait_window(config)

    # If an exception is raised in the config window, closing it before raising
    except (Exception,) as exc:
      # Not much we can do if there's no logger set to report Exception
      if self._logger is not None:
        self._logger.exception("Caught exception in the configuration "
                               "window !", exc_info=exc)
      if config is not None:
        config.stop()
      raise CameraConfigError
    # Special case of KeyboardInterrupt because it's a non-local exception
    except KeyboardInterrupt:
      if config is not None:
        config.stop()
      raise

    # Getting the image dtype and shape for setting the shared Array
    if config.shape is not None:
      if self._img_shape is not None and self._img_shape != config.shape:
        self.log(logging.WARNING, f"The img_shape from the configuration "
                                  f"window {config_class.__name__} "
                                  f"({config.shape}) is different from the "
                                  f"existing one ({self._img_shape}), setting "
                                  f"it anyway to the new value")
      self._img_shape = config.shape
    if config.dtype is not None:
      if self._img_dtype is not None and self._img_dtype != str(config.dtype):
        self.log(logging.WARNING, f"The img_dtype from the configuration "
                                  f"window {config_class.__name__} "
                                  f"({config.dtype}) is different from the "
                                  f"existing one ({self._img_dtype}), setting "
                                  f"it anyway to the new value")
      self._img_dtype = str(config.dtype)

    return config.get_config()

  def default_configuration(self) -> None:
    """Runs the generic Camera configuration window.

    The selected image shape and dtype are stored on this Block for creation
    of its shared image buffer. No configuration response is sent to a
    downstream Block.

    Raises:
      RuntimeError: If the Camera has not been initialized.
      CameraConfigError: If the configuration window fails.
      KeyboardInterrupt: If configuration is interrupted by the user.
    """

    if self._camera is None:
      raise RuntimeError("Cannot start the configuration window because the "
                         "Camera wasn't defined")

    self.log(logging.INFO, "Displaying the base configuration window")
    self.configure(self._camera, CameraConfig)
    self.log(logging.INFO, "Camera configuration done")

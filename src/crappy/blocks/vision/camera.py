# coding: utf-8

from collections.abc import Callable
import numpy as np
from collections import defaultdict
import logging
from time import time, strftime, gmtime
from types import MethodType
from typing import Any

from .block import VisionBlock
from ..camera import deprecated_cameras, camera_dict, DummyCam
from ...tool.camera_config import CameraConfig
from ...camera import Camera as BaseCam
from ..._global import CameraConfigError, PrepareError


class CameraSource(VisionBlock):
  """This :class:`~crappy.blocks.vision.VisionBlock` can drive a
  :class:`~crappy.camera.Camera` object. It can acquire images and send them to
  one or several downstream visionBlocks. It can only drive one Camera.

  It takes no input :class:`~crappy.links.Link` in a majority of situations,
  and usually doesn't have output Links neither. The only situations when it
  can accept input Links is when an ``image_generator`` is defined, or when
  defining a ``software_trig_label``. Each time an image is sent through the
  downstream :class:`~crappy.links.ImageLink`, a message is also sent through
  the downstream :class:`~crappy.links.Link` containing the timestamp, the
  image index, and the metadata. They are respectively carried by the `'t(s)'`,
  `'img_index'` and `'meta'` labels. This is useful for performing an action
  conditionally at each new acquired image.

  Before a test starts, this Block can display a
  :class:`~crappy.tool.camera_config.CameraConfig` window in which the user can
  visualize the acquired images, and interactively tune all the
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` available
  for the instantiated :class:`~crappy.camera.Camera`. Depending on the nature
  of the Blocks linked to this one with :class:`~crappy.links.ImageLink`, one
  or more specialized configuration windows can be opened.

  Note:
    This Block is only in charge of the image acquisition, it has to be linked
    to other :class:`~crappy.blocks.vision.VisionBlock` for images to be
    processed, saved, displayed, etc.

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
    """Sets the arguments and initializes the parent class.

    Args:
      camera: The name of the :class:`~crappy.camera.Camera` object to use for
        acquiring the images. Arguments can be passed to this Camera as
        ``kwargs`` of this Block. This argument is ignored if the
        ``image_generator`` argument is provided.
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output. Allows applying a post-processing
        operation to the acquired images. This is done right after the
        acquisition, so the original image is permanently lost and only the
        transformed image is passed on. The transform operation is not
        parallelized, so it might negatively affect the acquisition framerate
        if it is too heavy.
      config: If :obj:`True`, a
        :class:`~crappy.tool.camera_config.CameraConfig` window is displayed
        before the test starts. There, the user can interactively adjust the
        different
        :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
        available for the selected :class:`~crappy.camera.Camera`, and
        visualize the acquired images. The test starts when closing the
        configuration window. If not enabled, the ``img_dtype`` and
        ``img_shape`` arguments must be provided. The type of configuration
        window that is opened depends on the nature of the downstream Blocks
        in the graph of ImageLinks. More than one configuration window can be
        opened if multiple downstream blocks have different requirements.
      image_generator: A callable taking two :obj:`float` as arguments and
        returning an image as a :obj:`numpy.array`. **This argument is intended
        for use in the examples of Crappy, to apply an artificial strain on a
        base image. Most users should ignore it.** When given, the ``camera``
        argument is ignored and the images are acquired from the generator. To
        apply a strain on the image, strain values (in `%`) should be sent to
        the Camera Block over the labels ``'Exx(%)'`` and ``'Eyy(%)'``.
      software_trig_label: The name of a label used as a software trigger for
        the :class:`~crappy.camera.Camera`. If given, images will only be
        acquired when receiving data over this label. The received value does
        not matter. This software trigger is not meant to be very precise, it
        is recommended not to rely on it for a trigger frequency greater than
        10Hz, in which case a hardware trigger should be preferred if available
        on the camera.
      img_shape: The shape of the images that this Block sends to downstream
        Blocks. Set to :obj:`None` if it doesn't output images. The shape
        should be given as a :obj:`tuple` of :obj:`int`, as returned by
        :obj:`numpy.shape`. **This argument is mandatory in case the Block
        doesn't have a configuration window/mechanism.** If a configuration is
        used, the value of this argument is ignored.
      img_dtype: The dtype of the images that this Block sends to downstream
        Blocks. Set to :obj:`None` if it doesn't output images. The dtype
        should be given as a :obj:`str`, as returned by :obj:`numpy.dtype`.
        **This argument is mandatory in case the Block doesn't have a
        configuration window/mechanism.** If a configuration is used, the value
        of this argument is ignored.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.
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

    # Checking if the requested camera exists in Crappy
    if image_generator is None:
      if camera not in camera_dict:
        possible = ', '.join(sorted(camera_dict.keys()))
        raise ValueError(f"Unknown Camera type: {camera}! "
                         f"The possible types are: {possible}")
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
    """Opens the :class:`~crappy.camera.Camera` and displays the configuration
    GUI.

    This method calls the :meth:`crappy.camera.Camera.open` method of the
    :class:`~crappy.camera.Camera` object.
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
        """Method generating the frames using the ``image_generator`` argument
        if one was provided."""

        return time(), self_.apply_soft_roi(self._image_generator(self_.Exx,
                                                                  self_.Eyy))

      self._camera.get_image = MethodType(get_image, self._camera)

    # Instantiating the Camera object for acquiring the images
    else:
      self._camera = camera_dict[self._camera_name]()
      self.log(logging.INFO, f"Opening the {self._camera_name} Camera")
      if self._camera is None:
        raise RuntimeError("The Camera wasn't set whereas it should be")
      self._camera.open(**self._camera_kwargs)
      self.log(logging.INFO, f"Opened the {self._camera_name} Camera")

    if (self.config_requests_in and
        (not self._config_cam or not self._allow_downstream_config)):
      unhandled = [request.requester for request in self.config_requests_in]
      raise RuntimeError(f"The combination of the config and "
                         f"allow_downstream_config arguments is leading to "
                         f"unhandled config requests from Blocks "
                         f"{', '.join(unhandled)}, aborting early!\nTry to "
                         f"set config=True and allow_downstream_config=True, "
                         f"or provide configuration information to the "
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
        self.default_configuration()

    else:
      self.log(logging.INFO, "Skipping interactive configuration as requested")

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
    """This method receives data from upstream Blocks, acquires a frame from
    the :class:`~crappy.camera.Camera` object, and transmits it to all the
    downstream Blocks.

    The frame is sent through the output :class:`~crappy.links.ImageLink`, and
    a message containing information on each acquired image is sent through the
    :class:`~crappy.links.Link` if any. This message contains: on label 't(s)'
    the time of the acquisition, on label 'img_index' the unique image ID, and
    on label 'meta' the complete metadata dictionary.

    The image is acquired by calling the
    :meth:`~crappy.camera.Camera.get_image` method of the Camera object. If
    only a timestamp is returned by this method, and not a complete :obj:`dict`
    of metadata, some basic metadata is generated here and transmitted to the
    CameraProcesses.

    This method also manages the software trigger if this option was set,
    applies the image transformation function if one was given, and displays
    the FPS of the acquisition if required.
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
    """This method stops the image acquisition on the
    :class:`~crappy.camera.Camera`.

    For stopping the image acquisition, the :meth:`~crappy.camera.Camera.close`
    method is called.
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
    """Runs one interactive configuration request for a Camera.

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
    of its shared image buffers. No configuration response is sent to a
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

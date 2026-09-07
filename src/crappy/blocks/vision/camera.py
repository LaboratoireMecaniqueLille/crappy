# coding: utf-8

from collections.abc import Callable
import numpy as np
from collections import defaultdict
import logging
from time import time, strftime, gmtime
from types import MethodType

from .block import VisionBlock
from ..camera import deprecated_cameras, camera_dict, DummyCam
from ...tool.camera_config import CameraConfig
from ...camera import Camera as BaseCam
from ..._global import CameraConfigError

"""
Argument lets source know it has to use a specific configuration window
Otherwise, rely on a graph, that is anyway generated to avoid cyclicity
Maybe the extra argument can override the graph finding
"""


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
               image_generator: Callable[[float, float],
                                         np.ndarray] | None = None,
               software_trig_label: str | None = None,
               img_shape: tuple[int, int] | tuple[int, int, int] | None = None,
               img_dtype: str | None = None,
               debug: bool | None = False,
               freq: float | None = 100,
               display_freq: bool = False,
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
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      **kwargs: Any additional argument will be passed to the
        :class:`~crappy.camera.Camera` object, and used as a kwarg to its
        :meth:`~crappy.camera.Camera.open` method.
    """

    super().__init__(img_shape=img_shape,
                     img_dtype=img_dtype,
                     debug=debug,
                     freq=freq,
                     display_freq=display_freq)

    self._camera: BaseCam | None = None

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
      self._camera_name = camera
    else:
      self._camera_name = 'Image Generator'

    # Incrementing the count of opened cameras for this camera type
    CameraSource.cam_count[self._camera_name] += 1

    # Validate arguments before setting them
    if (software_trig_label is not None and
        (not isinstance(software_trig_label, str) or not software_trig_label)):
      raise ValueError("If provided, software_trig_label must be a non-empty "
                       "string")
    if config is not None and not isinstance(config, bool):
      raise ValueError("If provided, config should be a boolean")
    if transform is not None and not callable(transform):
      raise ValueError("If provided, transform should be a callable")
    if image_generator is not None and not callable(image_generator):
      raise ValueError("If provided, image_generator should be a callable")

    # Setting the other attributes
    self._trig_label: str | None = software_trig_label
    self._config_cam: bool = config
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

    # Displaying the configuration window if required
    if self._config_cam:
      self.log(logging.INFO, "Displaying the configuration window")
      self.configure()
      self.log(logging.INFO, "Camera configuration done")

    # Setting the camera to 'Hardware' trig if it's in 'Hdw after config' mode
    if (self._camera.trigger_name in self._camera.settings and
        getattr(self._camera,
                self._camera.trigger_name) == 'Hdw after config'):
      self.log(logging.INFO, "Setting the trigger mode to Hardware")
      setattr(self._camera, self._camera.trigger_name, 'Hardware')

    # Ensuring a dtype and a shape were given for the image
    if self._img_dtype is None or self._img_shape is None:
      raise ValueError(f"Cannot launch the Camera processes for camera "
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
      self._display_freq()

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

  def configure(self) -> None:
    """This method should instantiate and start a
    :class:`~crappy.tool.camera_config.CameraConfig` window for configuring the
    :class:`~crappy.camera.Camera` object.

    It should also handle the case when an exception is raised in the
    configuration window.

    It decides which specialized type(s) of
    :class:`~crappy.tool.camera_config.CameraConfig` window(s) to open based on
    the analysis of the :class:`~crappy.blocks.vision.VisionBlock` linking
    graph.
    """

    config = None

    # Instantiating and starting the configuration window
    try:
      if self._camera is None:
        raise RuntimeError("Cannot start the configuration window because the "
                           "Camera wasn't defined")
      if self._log_queue is None:
        raise RuntimeError("Cannot start the configuration window because the "
                           "log_queue wasn't defined")
      config = CameraConfig(self._camera, self._log_queue,
                            self._log_level, self.freq)
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

    # Getting the image dtype and shape for setting the shared Array
    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = str(config.dtype)

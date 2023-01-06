# coding: utf-8

from typing import Callable, Union, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from time import time, strftime, gmtime
from re import fullmatch
from types import MethodType
import logging

from .block import Block
from .displayer import Displayer
from ..camera import camera_list, Camera as BaseCam
from ..tool import Camera_config, Box
from .._global import OptionalModule

try:
  import SimpleITK as Sitk
except (ModuleNotFoundError, ImportError):
  Sitk = OptionalModule("SimpleITK")

try:
  import PIL
except (ModuleNotFoundError, ImportError):
  PIL = OptionalModule("Pillow")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Camera(Block):
  """This block simply acquires images from a camera.

  It can then save the images, and / or display them. The image acquisition can
  be triggered via incoming links. Optionally, a configuration window can be
  displayed for interactively tuning the camera settings before the test
  starts.

  This class also serves as a base class for other blocks that perform image
  processing on the acquired frames.
  """

  cam_count = dict()

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: bool = False,
               freq: float = 200,
               debug: bool = False,
               save_images: bool = False,
               img_name: str = "{self._n_loops:6d}_{t-self.t0:.6f}.tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               **kwargs) -> None:
    """Sets the args and initializes the parent class.

    Args:
      camera: The name of the camera to control. See :ref:`Cameras` for an
        exhaustive list of available cameras.
      transform: A function taking an image as an argument and returning a
        transformed image. The original image is discarded and only the
        transformed one is kept for processing, display and saving.
      config: If :obj:`True`, a config window is shown before the test starts
        for interactively tuning the camera settings. It also allows selecting
        the spots to track.
      display_images: If :obj:`True`, a window displays the acquired images
        in low resolution during the test. This display is mainly intended for
        debugging and visual follow-up, but not for displaying high-quality
        images.
      displayer_backend: If ``display_images`` is :obj:`True`, the backend to
        use for the display window. Should be one of :
        ::

          'cv2', 'mpl'

        If not given, OpenCV will be used if available.
      displayer_framerate: If ``display_images`` is :obj:`True`, sets the
        maximum framerate for updating the display window. This setting allows
        limiting the resources used by the displayer. Note that the actual
        achieved framerate might differ, this is just the maximum limit.
      software_trig_label: If given, the block will only acquire images when
        receiving data on this label. The received data can be anything, even
        empty. This label will thus de facto act as a software trigger for the
        camera.
      verbose: If :obj:`True`, the achieved framerate will be displayed in the
        console during the test.
      freq: If given, the block will try to loop at this frequency. If it is
        lower than the framerate of the camera, frames will be dropped. This
        argument can be used for limiting the achieved framerate when the
        camera doesn't support framerate control.
      save_images: If :obj:`True`, the acquired images are saved on the
        computer during the test. Note that saving images uses CPU, so the
        achieved performance might drop when this feature is in use.
      img_name: If ``save_images`` is :obj:`True`, the template for naming the
        recorded images. It is evaluated as an `f-string`, and must contain the
        file extension at the end. For building the `f-string`, the
        ``self._n_loops`` attribute holds the loop number, and ``t-self.t0``
        holds the current timestamp.
      save_folder: If ``save_images`` is :obj:`True`, the directory to save
        images to. If it doesn't exist, it will be created. If not given, the
        images are saved in a folder named `Crappy_images` and created next to
        the file being run.
      save_period: If ``save_images`` is :obj:`True`, only one every this
        number of images will be saved.
      save_backend: The backend to use for saving the images. Should be one
        of :
        ::

          'sitk', 'pil', 'cv2'

        If not specified, SimpleITK will be used if available, then OpenCV as a
        second choice, and finally Pillow if none of the others was available.
      image_generator: A function taking two floats as arguments, and returning
        an image. It is only used for demonstration without camera in the
        examples, and isn't meant to be used in an actual test.
      **kwargs: Any additional argument to pass to the camera.
    """

    super().__init__()

    self.verbose = verbose
    self.freq = freq
    self.niceness = -10
    self.log_level = logging.DEBUG if debug else logging.INFO

    self._camera: Optional[BaseCam] = None
    self._displayer: Optional[Displayer] = None

    # Checking if the requested camera exists in Crappy
    if image_generator is None:
      if camera.capitalize() not in camera_list:
        raise ValueError(f"No camera named {camera.capitalize()} found in the "
                         f"list of available cameras !")
      self._camera_name = camera.capitalize()
    else:
      self._camera_name = 'Image Generator'

    # Counting the number of instantiated cameras for each type
    if self._camera_name not in Camera.cam_count:
      Camera.cam_count[self._camera_name] = 1
    else:
      Camera.cam_count[self._camera_name] += 1

    # Trying the different possible backends and checking if the given one
    # is correct
    if save_images:
      if save_backend is None:
        if not isinstance(Sitk, OptionalModule):
          self._save_backend = 'sitk'
        elif not isinstance(cv2, OptionalModule):
          self._save_backend = 'cv2'
        elif not isinstance(PIL, OptionalModule):
          self._save_backend = 'pil'
        else:
          raise ModuleNotFoundError("Neither SimpleITK, opencv-python nor "
                                    "Pillow could be imported, no backend "
                                    "found for saving the images")
      elif save_backend in ('sitk', 'pil', 'cv2'):
        self._save_backend = save_backend
      else:
        raise ValueError("The save_backend argument should be either 'sitk', "
                         "'pil' or 'cv2' !")
    else:
      self._save_backend = None

    # Checking that the given image name is valid
    if save_images:
      if fullmatch(r'.*\{.+}.*', img_name) is None:
        raise ValueError("img_name cannot be evaluated as a regular "
                         "expression !")
      elif fullmatch(r'.+\..+', img_name) is None:
        raise ValueError("No extension given in the img_name argument !")
      self._img_name = img_name
    else:
      self._img_name = None

    # Setting a default save folder if not given
    if save_images:
      if save_folder is None:
        self._save_folder = Path.cwd() / 'Crappy_images'
      else:
        self._save_folder = Path(save_folder)
    else:
      self._save_folder = None

    self._save_period = int(save_period)

    # Instantiating the displayer window if requested
    if display_images:
      self._displayer = Displayer(f"Displayer {camera} "
                                  f"{Camera.cam_count[self._camera_name]}",
                                  displayer_framerate,
                                  displayer_backend)

    # Setting the other attributes
    self._save_images = save_images
    self._trig_label = software_trig_label
    self._config_cam = config
    self._transform = transform
    self._image_generator = image_generator
    self._camera_kwargs = kwargs

    self._n_loops = 0

  def prepare(self) -> None:
    """Preparing the save folder, opening the camera and displaying the
    configuration GUI."""

    # Creating the folder for saving the images if it doesn't exist
    if self._save_folder is not None:
      if not self._save_folder.exists():
        Path.mkdir(self._save_folder, exist_ok=True, parents=True)

    # Case when the images are generated and not acquired
    if self._image_generator is not None:
      self._camera = BaseCam()
      self._camera.add_scale_setting('Exx', -100., 100., None, None, 0.)
      self._camera.add_scale_setting('Eyy', -100., 100., None, None, 0.)
      self._camera.set_all()

      def get_image(self_) -> Tuple[float, np.ndarray]:
        return time(), self._image_generator(self_.Exx, self_.Eyy)

      self._camera.get_image = MethodType(get_image, self._camera)

    # Case when an actual camera object is responsible for acquiring the images
    else:
      self._camera = camera_list[self._camera_name]()
      self._camera.open(**self._camera_kwargs)

    if self._config_cam:
      config = Camera_config(self._camera)
      config.main()

    if self._displayer is not None:
      self._displayer.prepare()

  def loop(self) -> None:
    """Receives the incoming data, acquires an image, displays it, saves it,
    and finally processes it if needed."""

    data = self.recv_last_data(fill_missing=False)

    # Waiting for the trig label if it was given
    if self._trig_label is not None and self._trig_label not in data:
      return

    # Updating the image generator if there's one
    if self._image_generator is not None:
      if 'Exx(%)' in data:
        self._camera.Exx = data['Exx(%)']
      if 'Eyy(%)' in data:
        self._camera.Eyy = data['Eyy(%)']

    # Actually getting the image from the camera object
    ret = self._camera.get_image()
    if ret is None:
      return
    metadata, img = ret

    # Building the metadata if it was not provided
    if isinstance(metadata, float):
      metadata = {'t(s)': metadata,
                  'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S",
                                               gmtime(metadata)),
                  'SubsecTimeOriginal': f'{metadata % 1:.6f}',
                  'ImageUniqueID': self._n_loops}
    metadata['t(s)'] -= self.t0

    self._n_loops += 1

    # Applying the transform function
    if self._transform is not None:
      img = self._transform(img)

    # Updating the displayer
    if self._displayer is not None:
      self._displayer.update(img)

    # Saving the image
    if self._save_images and not self._n_loops % self._save_period:
      path = str(self._save_folder / eval('f"{}"'.format(self._img_name)))
      self._save(img, path)

    # Performing the additional actions for subclasses
    self._additional_loop(metadata, img)

  def finish(self) -> None:
    """Closes the camera and the displayer."""

    if self._image_generator is None and self._camera is not None:
      self._camera.close()

    if self._displayer is not None:
      self._displayer.finish()

  def _save(self, img: np.ndarray, path: str) -> None:
    """Simply saves the given image to the given path using the selected
    backend."""

    if self._save_backend == 'sitk':
      Sitk.WriteImage(Sitk.GetImageFromArray(img), path)

    elif self._save_backend == 'cv2':
      cv2.imwrite(path, img)

    elif self._save_backend == 'pil':
      PIL.Image.fromarray(img).save(path)

  def _additional_loop(self, meta: Dict[str, Any], img: np.ndarray) -> None:
    """Additional action to perform in the loop, used by subclasses of the
    Camera block."""

    ...

  @staticmethod
  def _draw_box(img: np.ndarray, box: Box) -> None:
    """Draws a box on top of an image."""

    if box.no_points():
      return

    x_top, x_bottom, y_left, y_right = box.sorted()

    for line in ((box.y_start, slice(x_top, x_bottom)),
                 (box.y_end, slice(x_top, x_bottom)),
                 (slice(y_left, y_right), x_top),
                 (slice(y_left, y_right), x_bottom),
                 (box.y_start + 1, slice(x_top, x_bottom)),
                 (box.y_end - 1, slice(x_top, x_bottom)),
                 (slice(y_left, y_right), x_top + 1),
                 (slice(y_left, y_right), x_bottom - 1)
                 ):
      img[line] = 255 * int(np.mean(img[line]) < 128)

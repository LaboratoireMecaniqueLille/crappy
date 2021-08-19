# coding: utf-8

from sys import platform
import os

from .block import Block
from ..camera import camera_list
from ..tool import Camera_config
from .._global import OptionalModule

try:
  import SimpleITK as Sitk
except (ModuleNotFoundError, ImportError):
  Sitk = None

try:
  import PIL
except (ModuleNotFoundError, ImportError):
  PIL = None

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Camera(Block):
  """Reads images from a camera object, saves and/or sends them to another
  block.

  It can be triggered by an other block, internally, or try to run at a given
  framerate.
  """

  def __init__(self,
               camera,
               save_folder=None,
               verbose=False,
               labels=None,
               fps_label=False,
               img_name="{self.loops:06d}_{t-self.t0:.6f}",
               ext='tiff',
               save_period=1,
               save_backend=None,
               transform=None,
               input_label=None,
               config=True,
               **kwargs):
    """Sets the args and initializes parent class.

    Args:
      camera (:obj:`str`): The name of the camera to control. See
        :ref:`Cameras` for an exhaustive list of available ones.
      save_folder (:obj:`str`, optional): The directory to save images to. If
        it doesn't exist it will be created. If :obj:`None` the images won't be
        saved.
      verbose (:obj:`bool`, optional): If :obj:`True`, the block will print the
        number of `loops/s`.
      labels (:obj:`list`, optional): Names of the labels for respectively time
        and the frame.
      fps_label (:obj:`str`, optional): If set, ``self.max_fps`` will be set to
        the value received by the block with this label.
      img_name (:obj:`str`, optional): Template for the name of the image to
        save. It is evaluated as an `f-string`.
      ext (:obj:`str`, optional): Extension of the image. Make sure it is
        supported by the saving backend.
      save_period (:obj:`int`, optional): Will save only one in `x` images.
      save_backend (:obj:`str`, optional): Module to use to save the images.
        The supported backends are: :mod:`sitk` (SimpleITK), :mod:`cv2`
        (OpenCV) and :mod:`pil` (Pillow). If :obj:`None`, will try :mod:`sitk`
        and then :mod:`cv2` if not successful.
      transform (:obj:`function`, optional): Function to be applied on the
        image before sending. It will not be applied on the saved images.
      input_label (:obj:`str`, optional): If specified, the image will not be
        read from a camera object but from this label.
      config (:obj:`bool`, optional): Show the popup for config ?
      **kwargs: Any additional specific argument to pass to the camera.
    """

    Block.__init__(self)
    self.niceness = -10
    self.save_folder = save_folder
    self.verbose = verbose
    self.labels = ['t(s)', 'frame'] if labels is None else labels
    self.fps_label = fps_label
    self.img_name = img_name
    self.ext = ext
    self.save_period = save_period
    self.save_backend = save_backend
    self.transform = transform
    self.input_label = input_label
    self.config = config

    self.camera_name = camera.capitalize()
    self.cam_kw = kwargs
    assert self.camera_name in camera_list or self.input_label,\
        "{} camera does not exist!".format(self.camera_name)
    if self.save_backend is None:
      if Sitk is None:
        self.save_backend = "cv2"
      else:
        self.save_backend = "sitk"
    assert self.save_backend in ["cv2", "sitk", "pil"],\
        "Unknown saving backend: " + self.save_backend
    self.save = getattr(self, "save_" + self.save_backend)
    self.loops = 0
    self.t0 = 0

  def prepare(self, send_img=True):
    sep = '\\' if 'win' in platform else '/'
    if self.save_folder and not self.save_folder.endswith(sep):
      self.save_folder += sep
    if self.save_folder and not os.path.exists(self.save_folder):
      try:
        os.makedirs(self.save_folder)
      except OSError:
        assert os.path.exists(self.save_folder),\
            "Error creating " + self.save_folder
    self.ext_trigger = bool(
        self.inputs and not (self.fps_label or self.input_label))
    if self.input_label is not None:
      # Exception to the usual inner working of Crappy:
      # We receive data from the link BEFORE the program is started
      self.ref_img = self.inputs[0].recv()[self.input_label]
      self.camera = camera_list['Camera']()
      self.camera.max_fps = 30
      self.camera.get_image = lambda: (0, self.ref_img)
      return
    self.camera = camera_list[self.camera_name]()
    self.camera.open(**self.cam_kw)
    if self.config:
      conf = Camera_config(self.camera)
      conf.main()
    # Sending the first image before the actual start
    if send_img:
      t, img = self.get_img()
      self.send([0, img])

  @staticmethod
  def save_sitk(img, fname):
    image = Sitk.GetImageFromArray(img)
    Sitk.WriteImage(image, fname)

  @staticmethod
  def save_cv2(img, fname):
    cv2.imwrite(fname, img)

  @staticmethod
  def save_pil(img, fname):
    PIL.Image.fromarray(img).save(fname)

  def get_img(self):
    """Waits the appropriate time/event to read an image, reads it, saves it if
    asked to, applies the transformation and increases counter."""

    if self.input_label:
      data = self.inputs[0].recv()
      return data['t(s)']+self.t0, data[self.input_label]
    if not self.ext_trigger:
      if self.fps_label:
        while self.inputs[0].poll():
          self.camera.max_fps = self.inputs[0].recv()[self.fps_label]
      t, img = self.camera.read_image()  # NOT constrained to max_fps
    else:
      data = self.inputs[0].recv()  # wait for a signal
      if data is None:
        return
      t, img = self.camera.get_image()  # self limiting to max_fps
    self.loops += 1
    if self.save_folder and self.loops % self.save_period == 0:
      self.save(img, self.save_folder +
          eval('f"{}"'.format(self.img_name)) + f".{self.ext}")
    if self.transform:
      img = self.transform(img)
    return t, img

  def loop(self):
    t, img = self.get_img()
    self.send([t-self.t0, img])

  def finish(self):
    if self.input_label is None:
      self.camera.close()

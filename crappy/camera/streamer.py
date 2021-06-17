# coding: utf-8

from time import time, sleep
import re
from glob import glob
from .._global import OptionalModule
try:
  import SimpleITK as Sitk
except (ModuleNotFoundError, ImportError):
  Sitk = None
  try:
    import cv2
  except (ModuleNotFoundError, ImportError):
    cv2 = OptionalModule("opencv-python")

from .camera import Camera
from .._global import CrappyStop


class Streamer(Camera):
  """This is a fake sensor meant to stream images that were already saved.

  Note:
    It needs a way to locate the time of each frame in the name of the picture.
    This is done using regular expressions.

    :meth:`__init__` takes no args, the arguments must be given when calling
    :meth:`open` (like all cameras).
  """

  def __init__(self):
    Camera.__init__(self)
    self.frame = 0

  def open(self,
           path,
           pattern="img_\\d+_(\\d+\\.\\d+)\\.tiff",
           start_delay=0,
           modifier=lambda img: img):
    """Sets the instance arguments.

    Args:
      path (:obj:`str`): The path of the folder containing the images.
      pattern (:obj:`str`, optional): The regular expression matching the
        images and returning the time.

        Note:
          `"\\d"` matches digits, `"\\d+"` matches a group of digits. `()` is a
          capturing group, returning what is inside. Dot is a special character
          and needs to be escaped. The default value is compatible with the
          naming method of the :ref:`Camera` and :ref:`Videoextenso` blocks.

      start_delay (:obj:`float`, optional): Before actually streaming the image
        flux you can set a delay in seconds during which the first image will
        be streamed in a loop.

        Note:
          This can be useful to give time for spot selection when using
          videoextenso.

      modifier: To apply a function to the image before sending it.
    """

    self.modifier = modifier
    pattern = "^" + path + pattern + "$"
    regex = re.compile(pattern)
    files = glob(path + "*")
    img_list = [f for f in files if regex.match(f)]
    assert img_list, "No matching image found!"
    print("[image streamer]", len(img_list), "images to stream")
    self.img_dict = {}
    self.time_table = []
    for img in img_list:
      t = float(regex.match(img).groups()[0])
      self.img_dict[t] = img
      self.time_table.append(t)
    self.time_table.sort()
    print("[image streamer] Duration:", self.time_table[-1], "s")
    self.t0 = time() + start_delay

  def close(self):
    self.frame = 0
    self.img_dict = {}
    self.time_table = []

  def get_image(self):
    if self.frame == len(self.time_table):
      raise CrappyStop
    img_t = self.time_table[self.frame]
    if Sitk is not None:
      img = self.modifier(
        Sitk.GetArrayFromImage(Sitk.ReadImage(self.img_dict[img_t])))
    else:
      img = self.modifier(cv2.imread(self.img_dict[img_t], 0))
    t = time()
    delay = self.time_table[self.frame] - t + self.t0
    if delay > 0:
      if t > self.t0:
        sleep(delay)
      else:
        return img_t, img
    # else:
    #   print("Streamer is", -1000 * delay, "ms late")
    self.frame += 1
    return img_t, img

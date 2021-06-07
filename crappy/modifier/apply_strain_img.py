import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

from .modifier import Modifier


class Apply_strain_img(Modifier):
  """This modifier reads the strain values along X and Y (in %) and creates an
  image deformed to match these values."""

  def __init__(self,
               img,
               exx_label: str = 'Exx(%)',
               eyy_label: str = 'Eyy(%)',
               img_label: str = 'frame') -> None:
    """Sets the instance attributes.

    Args:
      img: The image to use (must be a numpy array)
      exx_label (:obj:`str`, optional): The labels containing the strain to
        apply
      eyy_label (:obj:`str`, optional): The labels containing the strain to
        apply
      img_label (:obj:`str`, optional): The label of the generated image
    """

    self.img = img
    self.lexx = exx_label
    self.leyy = eyy_label
    self.img_label = img_label
    h, w = img.shape
    self.exx = np.concatenate((np.linspace(-w / 2, w / 2, w,
          dtype=np.float32)[np.newaxis, :],) * h, axis=0)
    self.eyy = np.concatenate((np.linspace(-h / 2, h / 2, h,
      dtype=np.float32)[:, np.newaxis],) * w, axis=1)
    xx, yy = np.meshgrid(range(w), range(h))
    self.xx = xx.astype(np.float32)
    self.yy = yy.astype(np.float32)

  def evaluate(self, d: dict) -> dict:
    exx, eyy = d[self.lexx] / 100, d[self.leyy] / 100
    tx, ty = (self.xx - (exx / (1 + exx)) * self.exx), \
             (self.yy - (eyy / (1 + eyy)) * self.eyy)
    d[self.img_label] = cv2.remap(self.img, tx, ty, 1)
    return d

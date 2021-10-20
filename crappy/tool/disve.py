# coding: utf-8

"""More documentation coming soon !"""

import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DISVE:
  def __init__(self,
               img0: np.ndarray,
               patches: list,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gditerations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               border: float = 0.1,
               safe: bool = True,
               follow: bool = True,
               show_image: bool = False) -> None:
    """Sets the disve parameters.

    Args:
      img0: Reference image for the DIC
      patches: Regions to track, should be a tuple of 4 values
        (pos x, pos y, height, width)
      alpha: Setting for disflow
      delta: Setting for disflow
      gamma: Setting for disflow
      finest_scale: Last scale for disflow (`0` means full scale)
      iterations: Variational refinement iterations
      gditerations: Gradient descent iterations
      patch_size: DIS patch size
      patch_stride: DIS patch stride
      border: Remove borders 10% of the size of the patch (`0.` to `0.5`)
    """

    self.img0 = img0
    self.patches = patches
    self.h, self.w = img0.shape
    self.alpha = alpha
    self.delta = delta
    self.gamma = gamma
    self.finest_scale = finest_scale
    self.iterations = iterations
    self.gditerations = gditerations
    self.patch_size = patch_size
    self.patch_stride = patch_stride
    self.border = border
    self.safe = safe
    self.follow = follow
    self.show_image = show_image

    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis.setGradientDescentIterations(self.gditerations)
    self.dis.setPatchSize(self.patch_size)
    self.dis.setPatchStride(self.patch_stride)
    self.offsets = [(0, 0) for _ in self.patches]
    self.adjust_offsets()
    if show_image:
      cv2.namedWindow('DISVE', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

  @staticmethod
  def get_patch(img: np.ndarray, patch: list,
                offset: tuple = (0, 0)) -> np.ndarray:
    ymin, xmin, h, w = patch
    ox, oy = offset
    return np.array(img[ymin + oy:ymin + h + oy,
                        xmin + ox:xmin + w + ox])

  def get_center(self, f: np.ndarray) -> np.ndarray:
    h, w, *_ = f.shape
    return f[int(h * self.border):int(h * (1 - self.border)),
             int(w * self.border):int(w * (1 - self.border))]

  def adjust_offsets(self):
    """
    Check if the patches are within the image

    If safe=True, will raise an error if a patch is out of bounds
    else, it will clamp it to the border of the image
    """
    for i, (patch, (ox, oy)) in enumerate(zip(self.patches, self.offsets)):
      ymin, xmin, h, w = patch
      if ox < -xmin:  # Left
        if self.safe:
          raise RuntimeError("Region exiting the ROI (left)")
        ox = -xmin
      elif ox > self.w - xmin - w:  # Right
        if self.safe:
          raise RuntimeError("Region exiting the ROI (right)")
        ox = self.w - xmin - w
      if oy < -ymin:  # Top
        if self.safe:
          raise RuntimeError("Region exiting the ROI (top)")
        oy = -ymin
      elif oy > self.h - ymin - h:  # Bottom
        if self.safe:
          raise RuntimeError("Region exiting the ROI (bottom)")
        oy = self.h - ymin - h

  def update_img(self, img: np.ndarray) -> None:
    for (ymin, xmin, h, w), (ox, oy) in zip(self.patches, self.offsets):
      img[ymin + oy:ymin + oy + 1, xmin + ox: xmin + ox + w] = 255
      img[ymin + h + oy:ymin + h + oy + 1, xmin + ox: xmin + ox + w] = 255
      img[ymin + oy: ymin + h + oy, xmin + ox:xmin + ox + 1] = 255
      img[ymin + oy: ymin + h + oy, xmin + h + ox:xmin + h + ox + 1] = 255
    cv2.imshow("DISVE", img)
    cv2.waitKey(5)

  def calc(self, img: np.ndarray) -> list:
    r = []
    for patch, offset in zip(self.patches, self.offsets):
      f = self.dis.calc(
          self.get_patch(self.img0, patch),
          self.get_patch(img, patch, offset), None)
      r.append(np.average(self.get_center(f), axis=(0, 1)).tolist())
    if self.follow:
      for disp, (ox, oy) in zip(r, self.offsets):
        disp[0] += ox
        disp[1] += oy
      self.offsets = [(round(ox), round(oy)) for ox, oy in r]
      self.adjust_offsets()
    if self.show_image:
      self.update_img(img)
    return sum(r, [])

  def close(self):
    if self.show_image:
      cv2.destroyWindow("DISVE")

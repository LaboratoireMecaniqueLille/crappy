# coding: utf-8

import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DISVE:
  def __init__(self,
               img0,
               patches,
               alpha=3,
               delta=1,
               gamma=0,
               finest_scale=1,
               iterations=1,
               gditerations=10,
               patch_size=8,
               patch_stride=3,
               border=0.1):
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

    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis.setGradientDescentIterations(self.gditerations)
    self.dis.setPatchSize(self.patch_size)
    self.dis.setPatchStride(self.patch_stride)
    self.last = [None for _ in self.patches]

  @staticmethod
  def get_patch(img, patch):
    ymin, xmin, h, w = patch
    return np.array(img[ymin:ymin + h, xmin:xmin + w])

  def get_center(self, f):
    h, w, *_ = f.shape
    return f[int(h * self.border):int(h * (1 - self.border)),
        int(w * self.border):int(w * (1 - self.border))]

  def calc(self, img):
    r = []
    for p in self.patches:
      f = self.dis.calc(
        self.get_patch(self.img0, p),
        self.get_patch(img, p), None)
      r.append(np.average(self.get_center(f), axis=(0, 1)).tolist())
    self.last = r
    return sum(r, [])

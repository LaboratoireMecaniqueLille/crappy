# coding: utf-8

"""More documentation coming soon !"""

from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
import numpy as np

from .fields import get_fields, Projector, get_res


class DISCorrel:

  def __init__(self,
               img0,
               bbox=None,
               fields=None,
               alpha=3,
               delta=1,
               gamma=0,
               finest_scale=1,
               init=True,
               iterations=1,
               gditerations=10,
               patch_size=8,
               patch_stride=3):
    """Sets the discorrel parameters.

        Args:
          img0:
          bbox:
          fields: Base of fields to use for the projection
          alpha: Setting for disflow
          delta: Setting for disflow
          gamma: Setting for disflow
          finest_scale: Last scale for disflow (`0` means full scale)
          init: If :obj:`True`, the last field is used to init
          iterations: Variational refinement iterations
          gditerations: Gradient descent iterations
          patch_size: DIS patch size
          patch_stride: DIS patch stride
        """

    self.img0 = img0
    self.h, self.w = img0.shape
    self.bbox = bbox
    self.fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    self.alpha = alpha
    self.delta = delta
    self.gamma = gamma
    self.finest_scale = finest_scale
    self.init = init
    self.iterations = iterations
    self.gditerations = gditerations
    self.patch_size = patch_size
    self.patch_stride = patch_stride
    if self.bbox is None:
      self.bbox = (0, 0, self.h, self.w)
    self.bh, self.bw = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
    self.fields = get_fields(self.fields, self.bh, self.bw)
    self.p = Projector(self.fields)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setGradientDescentIterations(self.gditerations)
    self.dis.setPatchSize(self.patch_size)
    self.dis.setPatchStride(self.patch_stride)
    self.dis_flow = np.zeros((self.h, self.w, 2))

  def crop(self, img):
    ymin, xmin, ymax, xmax = self.bbox
    return img[ymin:ymax, xmin:xmax]

  def calc(self, img):
    self.img = img
    if self.init:
      self.dis_flow = self.dis.calc(self.img0, img, self.dis_flow)
    else:
      self.dis_flow = self.dis.calc(self.img0, img, None)
    self.proj = self.p.get_scal(self.crop(self.dis_flow))
    return self.proj

  def dis_res(self):
    return get_res(self.img0, self.img, self.dis_flow)

  def dis_res_scal(self):
    return np.average(np.abs(get_res(self.img0, self.img, self.dis_flow)))

  def proj_flow(self):
    return self.p.get_full(self.dis_flow)

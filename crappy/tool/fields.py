# coding: utf-8

"""More documentation coming soon !"""

import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


def ones(h, w):
  return np.ones((h, w), dtype=np.float32)


def zeros(h, w):
  return np.zeros((h, w), dtype=np.float32)


Z = None


def z(h, w):
  global Z
  if Z is None or Z[0].shape != (h, w):
    sh = 1 / (w * w / h / h + 1) ** .5
    sw = w*sh/h
    Z = np.meshgrid(np.linspace(-sw, sw, w, dtype=np.float32),
                          np.linspace(-sh, sh, h, dtype=np.float32))
  return Z


def get_field(s, h, w):
  if s == 'x':
    return ones(h, w), zeros(h, w)
  elif s == 'y':
    return zeros(h, w), ones(h, w)
  elif s == 'r':
    u, v = z(h, w)
    # Ratio (angle) of the rotation
    # Should be π/180 to be 1 for 1 deg
    # Z has and amplitude of 1 in the corners
    # 360 because h²+w² is twice the distance center-corner
    r = (h ** 2 + w ** 2) ** .5 * np.pi / 360
    return v * r, -u * r
  elif s == 'exx':
    return (np.concatenate((np.linspace(-w / 200, w / 200, w,
            dtype=np.float32)[np.newaxis, :],) * h, axis=0),
            zeros(h, w))
  elif s == 'eyy':
    return (zeros(h, w),
            np.concatenate((np.linspace(-h / 200, h / 200, h,
            dtype=np.float32)[:, np.newaxis],) * w, axis=1))
  elif s == 'exy':
    return (np.concatenate((np.linspace(-h / 200, h / 200, h,
            dtype=np.float32)[:, np.newaxis],) * w, axis=1),
            zeros(h, w))
  elif s == 'eyx':
    return (zeros(h, w),
            np.concatenate((np.linspace(-w / 200, w / 200, w,
            dtype=np.float32)[np.newaxis, :],) * h, axis=0))
  elif s == 'exy2':
    return (np.concatenate((np.linspace(-h / 200, h / 200, h,
            dtype=np.float32)[:, np.newaxis],) * w, axis=1),
            (np.concatenate((np.linspace(-w / 200, w / 200, w,
            dtype=np.float32)[np.newaxis, :],) * h, axis=0)))

  elif s == 'z':
    u, v = z(h, w)
    # Zoom in %
    r = (h ** 2 + w ** 2) ** .5 / 200
    return u * r, v * r
  else:
    print("Unknown field:", s)
    raise NameError


def get_fields(l, h, w):
  r = np.empty((h, w, 2, len(l)), dtype=np.float32)
  for i, s in enumerate(l):
    if isinstance(s, np.ndarray):
      r[:, :, :, i] = s
    else:
      r[:, :, 0, i], r[:, :, 1, i] = get_field(s, h, w)
  return r


class Fielder:
  def __init__(self, flist, h, w):
    self.nfields = len(flist)
    self.h = h
    self.w = w
    fields = get_fields(flist, h, w)
    self.fields = [fields[:, :, :, i] for i in range(fields.shape[3])]

  def get(self, *x):
    return sum([i * f for i, f in zip(x, self.fields)])


class Projector:
  def __init__(self, base, check_orthogonality=True):
    if isinstance(base, list):
      self.base = base
    else:
      self.base = [base[:, :, :, i] for i in range(base.shape[3])]
    self.fielder = Fielder(self.base, *self.base[0].shape[:2])
    self.norms2 = [np.sum(b * b) for b in self.base]
    if check_orthogonality:
      from itertools import combinations
      s = []
      for a, b in combinations(self.base, 2):
        s.append(abs(np.sum(a * b)))
      maxs = max(s)
      if maxs / self.base[0].size > 1e-4:
        print("WARNING, base does not seem orthogonal!")
        print(s)

  def get_scal(self, flow):
    return [np.sum(vec * flow) / n2 for vec, n2 in zip(self.base, self.norms2)]

  def get_full(self, flow):
    return self.fielder.get(*self.get_scal(flow))


class OrthoProjector(Projector):
  def __init__(self, base):
    vec = [base[:, :, :, i] for i in range(base.shape[3])]
    new_base = [vec[0]]
    for v in vec[1:]:
      p = Projector(new_base, check_orthogonality=False)
      new_base.append(v - p.get_full(v))
    Projector.__init__(self, new_base)


def avg_ampl(f):
  return (np.sum(f[:, :, 0] ** 2 + f[:, :, 1] ** 2) / f.size * 2) ** .5


def remap(a, r):
  """Remaps `a` using given `r` the displacement as a result from
  correlation."""

  imy, imx = a.shape
  x, y = np.meshgrid(range(imx), range(imy))
  return cv2.remap(a.astype(np.float32),
      (x + r[:, :, 0]).astype(np.float32),
                   (y + r[:, :, 1]).astype(np.float32), 1)


def get_res(a, b, r):
  # return b - remap(a, -r)
  return a - remap(b, r)

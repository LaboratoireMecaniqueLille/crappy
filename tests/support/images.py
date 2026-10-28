"""Small synthetic images, with importable acquisition callbacks for spawn."""

import numpy as np


def zero_frame(exx, eyy):
  """Camera-wrapper input independent of strain or physical equipment."""

  return np.zeros((12, 16), dtype=np.uint8)


def gradient_image(minimum=0, maximum=255):
  x, y = np.mgrid[0:240, 0:320]
  return (minimum + (x + y) / np.max(x + y) * (maximum - minimum)).astype(np.uint8)


def spots_image():
  image = np.full((240, 320), 255, dtype=np.uint8)
  y, x = np.ogrid[:image.shape[0], :image.shape[1]]
  for x_center, y_center in ((80, 80), (80, 160), (160, 80), (160, 160)):
    image[(x - x_center) ** 2 + (y - y_center) ** 2 <= 20 ** 2] = 0
  return image

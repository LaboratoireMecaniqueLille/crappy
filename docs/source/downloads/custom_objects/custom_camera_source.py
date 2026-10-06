# coding: utf-8

# [custom-camera-start]
from tempfile import TemporaryDirectory
from time import time

import numpy as np

import crappy


# [custom-camera-class-start]
class GradientCamera(crappy.camera.Camera):

  def __init__(self) -> None:
    super().__init__()
    self.add_scale_setting(
        name='brightness',
        lowest=0,
        highest=254,
        default=40)
    self._width = 0
    self._height = 0
    self._frame = 0

  def open(self, width: int, height: int, brightness: int = 40) -> None:
    self._width = width
    self._height = height
    self._frame = 0
    self.set_all(brightness=brightness)

  def get_image(self) -> tuple[float, np.ndarray]:
    image = np.full(
        (self._height, self._width), self.brightness, dtype=np.uint8)
    image[:, self._frame % self._width] = 255
    self._frame += 1
    return time(), image

  def close(self) -> None:
    self._width = 0
    self._height = 0
# [custom-camera-class-end]


def main() -> None:
  with TemporaryDirectory(prefix='crappy_custom_camera_') as folder:
    # [custom-camera-use-start]
    camera = crappy.blocks.vision.CameraSource(
        camera='GradientCamera',
        config=False,
        img_shape=(48, 64),
        img_dtype='uint8',
        width=64,
        height=48,
        brightness=40,
        freq=20)
    # [custom-camera-use-end]

    recorder = crappy.blocks.vision.ImageRecorder(
        save_folder=folder,
        save_period=10,
        save_backend='npy',
        freq=20)
    reader = crappy.blocks.LinkReader(name='Saved frame', freq=10)
    stop = crappy.blocks.StopBlock('t(s) > 2')

    crappy.img_link(camera, recorder)
    crappy.link(camera, stop)
    crappy.link(recorder, reader)

    crappy.start()


if __name__ == '__main__':
  main()
# [custom-camera-end]

# coding: utf-8

# [custom-all-in-one-camera-start]
import numpy as np

import crappy


# [custom-all-in-one-camera-process-start]
class BrightestRowProcess(
    crappy.blocks.camera_processes.CameraProcess):

  def loop(self) -> None:
    if self.img is None or self.img.ndim != 2:
      raise ValueError('BrightestRowProcess requires greyscale images')

    row_means = np.mean(self.img, axis=1)
    self.send({
        't(s)': self.metadata['t(s)'],
        'img_index': self.metadata['ImageUniqueID'],
        'brightest_row(px)': int(np.argmax(row_means)),
    })
# [custom-all-in-one-camera-process-end]


# [custom-all-in-one-camera-block-start]
class BrightestRowCamera(crappy.blocks.Camera):

  def prepare(self) -> None:
    self.process_proc = BrightestRowProcess()
    super().prepare()
# [custom-all-in-one-camera-block-end]


def main() -> None:
  # [custom-all-in-one-camera-use-start]
  camera = BrightestRowCamera(
      'FakeCamera',
      config=False,
      img_shape=(120, 160),
      img_dtype='uint8',
      width=160,
      height=120,
      speed=25,
      fps=10,
      freq=20)
  # [custom-all-in-one-camera-use-end]

  reader = crappy.blocks.LinkReader(name='Brightest row', freq=10)
  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.link(camera, reader)
  crappy.link(camera, stop)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-all-in-one-camera-end]

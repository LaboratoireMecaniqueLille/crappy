# coding: utf-8

# [custom-vision-block-start]
import numpy as np

import crappy


# [custom-vision-block-class-start]
class BrightestRow(crappy.blocks.vision.VisionBlock):

  def __init__(self, freq: float = 30) -> None:
    super().__init__(freq=freq)

  def prepare(self) -> None:
    if len(self.img_inputs) != 1 or self.img_outputs:
      raise IOError(
          'BrightestRow requires one image input and no image output')
    super().prepare()

  def loop(self) -> None:
    updated_links = self.receive_imgs()
    if not updated_links:
      return

    received = self.last_received[updated_links[0]]
    if received.metadata is None:
      raise RuntimeError('Image metadata is missing')

    row_means = np.mean(received.img, axis=1)
    self.send({
        't(s)': received.metadata['t(s)'],
        'img_index': received.metadata['ImageUniqueID'],
        'brightest_row(px)': int(np.argmax(row_means)),
    })
# [custom-vision-block-class-end]


def main() -> None:
  camera = crappy.blocks.vision.CameraSource(
      'FakeCamera',
      config=False,
      img_shape=(120, 160),
      img_dtype='uint8',
      width=160,
      height=120,
      speed=25,
      fps=10,
      freq=20)

  # [custom-vision-block-use-start]
  brightness = BrightestRow(freq=30)

  crappy.img_link(camera, brightness, name='camera-images')
  # [custom-vision-block-use-end]

  reader = crappy.blocks.LinkReader(name='Brightest row', freq=10)
  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.link(brightness, reader)
  crappy.link(brightness, stop)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-vision-block-end]

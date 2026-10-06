# coding: utf-8

# [image-pipeline-start]
import crappy


def main() -> None:
  camera = crappy.blocks.vision.CameraSource(
      'FakeCamera',
      config=False,
      img_shape=(240, 320),
      img_dtype='uint8',
      width=320,
      height=240,
      speed=60,
      fps=20,
      freq=30)

  display = crappy.blocks.vision.ImageDisplayer(
      title='First VisionBlock pipeline',
      backend='mpl',
      framerate=10,
      freq=30)

  stop = crappy.blocks.StopBlock('t(s) > 5')

  crappy.img_link(camera, display)
  crappy.link(camera, stop)

  crappy.start()


if __name__ == '__main__':
  main()
# [image-pipeline-end]

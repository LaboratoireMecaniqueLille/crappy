# coding: utf-8

# [test-camera-object-start]
import crappy


def main() -> None:
  camera = crappy.camera.FakeCamera()

  try:
    camera.open(width=320, height=240, speed=60, fps=20)
    result = camera.get_image()
    if result is None:
      raise RuntimeError('The camera returned no image')

    timestamp, image = result
    print(f'Acquired image: {image.shape}, {image.dtype}')
    print(f'Timestamp: {timestamp}')
  finally:
    camera.close()


if __name__ == '__main__':
  main()
# [test-camera-object-end]

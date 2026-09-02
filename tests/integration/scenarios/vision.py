# coding: utf-8

from pathlib import Path

import numpy as np

import crappy
from crappy.blocks import Block


def generate_test_image(_: float, __: float) -> np.ndarray:
  """Returns a deterministic greyscale image for Camera recording."""

  y, x = np.indices((48, 64))
  return ((3 * x + 5 * y) % 256).astype(np.uint8)


def _make_speckle() -> np.ndarray:
  """Returns a reproducible textured image for correlation scenarios."""

  rng = np.random.default_rng(12345)
  return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


def build_camera_image_saver(output_dir: Path) -> tuple[Block, ...]:
  """Builds a generated Camera -> saved NPY images script."""

  camera = crappy.blocks.Camera(
    '',
    config=False,
    display_images=False,
    freq=30,
    save_images=True,
    save_folder=output_dir / 'images',
    save_period=1,
    save_backend='npy',
    image_generator=generate_test_image,
    img_shape=(48, 64),
    img_dtype='uint8')

  stop = crappy.blocks.StopBlock('t(s) > 1.2', freq=50)

  return camera, stop


def build_dicve_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a Generator -> DICVE -> Recorder script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 6,
      'condition': 'delay=1.0',
      'init_value': 0},),
    cmd_label='Exx(%)',
    spam=True,
    end_delay=0.2,
    freq=20)

  dicve = crappy.blocks.DICVE(
    '',
    config=False,
    display_images=False,
    freq=20,
    save_images=False,
    image_generator=crappy.tool.ApplyStrainToImage(_make_speckle()),
    img_shape=(128, 128),
    img_dtype='uint8',
    patches=((24, 24, 32, 32), (72, 72, 32, 32)),
    method='Parabola',
    safe=True,
    follow=False,
    raise_on_patch_exit=False)

  recorder = crappy.blocks.Recorder(
    output_dir / 'dicve.csv',
    labels=('t(s)', 'Eyy(%)', 'Exx(%)'),
    delay=0.1,
    freq=20)

  crappy.link(generator, dicve)
  crappy.link(dicve, recorder)

  return generator, dicve, recorder


def build_dis_correl_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a Generator -> DISCorrel -> Recorder script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 6,
      'condition': 'delay=1.0',
      'init_value': 0},),
    cmd_label='Exx(%)',
    spam=True,
    end_delay=0.2,
    freq=20)

  dis_correl = crappy.blocks.DISCorrel(
    '',
    config=False,
    display_images=False,
    freq=20,
    save_images=False,
    image_generator=crappy.tool.ApplyStrainToImage(_make_speckle()),
    img_shape=(128, 128),
    img_dtype='uint8',
    patch=(24, 24, 80, 80),
    fields=('exx', 'eyy'),
    labels=('t(s)', 'meta', 'Exx(%)', 'Eyy(%)'),
    finest_scale=1,
    iterations=1,
    gradient_iterations=5)

  recorder = crappy.blocks.Recorder(
    output_dir / 'dis_correl.csv',
    labels=('t(s)', 'Exx(%)', 'Eyy(%)'),
    delay=0.1,
    freq=20)

  crappy.link(generator, dis_correl)
  crappy.link(dis_correl, recorder)

  return generator, dis_correl, recorder

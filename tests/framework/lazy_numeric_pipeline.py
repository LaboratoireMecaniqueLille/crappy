"""Real numeric and image pipelines with unused optional imports forbidden."""

import importlib.abc
import multiprocessing as mp
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


FORBIDDEN = {'cv2', 'SimpleITK', 'PIL', 'scipy', 'skimage', 'tables',
             'tkinter', '_tkinter', 'PyQt6', 'pyqtgraph', 'matplotlib',
             'graphviz', 'gi', 'serial', 'paho', 'ximea', 'pycuda', 'nidaqmx'}


class RejectUnusedImports(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname.split('.')[0] in FORBIDDEN:
      raise AssertionError('Unused optional dependency: ' + fullname)


# Spawn/forkserver re-execute this module before reconstructing the Blocks.
sys.meta_path.insert(0, RejectUnusedImports())

import numpy as np

import crappy
from crappy.blocks import Generator, Sink
from crappy.blocks.vision import CameraSource, ImageRecorder
from crappy.tool.camera_config import Box, Overlay, SpotsBoxes, Zoom


def run(method):
  mp.set_start_method(method, force=True)
  try:
    with TemporaryDirectory(prefix='crappy-lazy-numeric-') as folder:
      generator = Generator([{'type': 'Constant', 'value': 1,
                              'condition': 'delay=0.3'}], debug=None)
      sink = Sink(debug=None)
      crappy.link(generator, sink)
      source = CameraSource('FakeCamera', config=False, width=5, height=4,
                             img_shape=(4, 5), img_dtype='uint8', debug=None)
      recorder = ImageRecorder(save_backend='npy', save_folder=folder, debug=None)
      crappy.img_link(source, recorder)
      assert isinstance(SpotsBoxes(Box()), SpotsBoxes)
      assert isinstance(Zoom(), Zoom) and isinstance(Overlay, type)
      crappy.start(log_level=None)
      images = sorted(Path(folder).rglob('*.npy'))
      assert images, 'The selected recording backend saved no frames'
      image = np.load(images[0])
      assert image.shape == (4, 5) and image.dtype == np.uint8
      assert not FORBIDDEN.intersection(sys.modules)
      assert not mp.active_children(), mp.active_children()
  finally:
    crappy.reset()


if __name__ == '__main__':
  run(sys.argv[1])

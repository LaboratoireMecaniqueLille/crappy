"""Run a discovered unittest group with required capabilities and deadlines."""

import argparse
from importlib import import_module
import math
import os
import runpy
import subprocess
import sys
import unittest

from tests.support.paths import PROJECT_ROOT


GROUPS = {
    'tests/unit': '',
    'tests/integration': '',
    'tests/documentation': '',
    'tests/gui/tkinter': 'tkinter',
    'tests/gui/pyqt': 'pyqt',
    'tests/gui/display/camera_processes': 'display',
    'tests/gui/display/vision': 'display',
}


def preserve_macos_tk(group: str) -> None:
  if sys.platform == 'darwin' and group in ('tkinter', 'display'):
    path = PROJECT_ROOT / '.github/scripts/run_macos_tk_tests.py'
    wrapper = runpy.run_path(str(path))
    wrapper['retain_first_tcl_interpreter']()


def check_capabilities(directory: str) -> None:
  """Probe installed imports and real windows in a disposable interpreter."""

  group = GROUPS[directory]
  modules = ['numpy', 'psutil']
  if directory != 'tests/documentation':
    modules.extend(('cv2', 'PIL.Image', 'skimage.filters', 'matplotlib'))
  if directory in ('tests/unit', 'tests/integration'):
    modules.extend(('SimpleITK', 'tables', 'graphviz'))
  if directory == 'tests/unit' or group == 'pyqt':
    modules.extend(('PyQt6.QtWidgets', 'pyqtgraph'))
  if group in ('tkinter', 'display'):
    modules.extend(('tkinter', 'PIL.ImageTk'))

  for name in modules:
    print(f'Checking {name}', flush=True)
    import_module(name)

  if group in ('tkinter', 'display'):
    import tkinter as tk

    root = tk.Tk()
    try:
      root.update()
    finally:
      root.destroy()
  if group == 'pyqt':
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication([])
    window = QWidget()
    try:
      window.show()
      app.processEvents()
    finally:
      window.close()
      app.processEvents()
      app.quit()
  if group == 'display':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    try:
      image = np.zeros((32, 32), dtype=np.uint8)
      cv2.imshow('Crappy display preflight', image)
      cv2.waitKey(1)
      figure = plt.figure()
      figure.canvas.draw()
      plt.show(block=False)
      plt.pause(0.01)
    finally:
      plt.close('all')
      cv2.destroyAllWindows()


def _cases(suite):
  for item in suite:
    if isinstance(item, unittest.TestSuite):
      yield from _cases(item)
    else:
      yield item


def run_suite(suite: unittest.TestSuite) -> int:
  """A successful CI group must execute tests after runtime capability skips."""

  collected = {(type(case), case.id()) for case in _cases(suite)}
  if not collected:
    print('No tests were collected', file=sys.stderr)
    return 1
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  # Skipped subtests do not mean their enclosing test was entirely skipped.
  skipped_tests = sum((type(test), test.id()) in collected
                      for test, _ in result.skipped)
  if result.testsRun <= skipped_tests:
    print('No tests executed without being skipped', file=sys.stderr)
    return 1
  return 0 if result.wasSuccessful() else 1


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', choices=GROUPS)
  parser.add_argument('--timeout', type=float, default=600,
                      help='test process deadline in seconds (default: 600)')
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument('--preflight', action='store_true', help=argparse.SUPPRESS)
  mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  if not math.isfinite(args.timeout) or args.timeout <= 0:
    parser.error('--timeout must be a finite positive number')
  group = GROUPS[args.directory]
  if group and os.environ.get('CRAPPY_TEST_GUI') != group:
    parser.error(f'{args.directory} requires CRAPPY_TEST_GUI={group}')

  if args.preflight or args.worker:
    preserve_macos_tk(group)
    if args.preflight:
      check_capabilities(args.directory)
      return 0
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / args.directory),
                            top_level_dir=str(PROJECT_ROOT))
    return run_suite(suite)

  # Neither the preflight nor a test may change the other's imported modules,
  # toolkit state, or first Tcl interpreter. Each owns a fresh process tree.
  from tests.support.processes import run_python

  environment = dict(os.environ, MPLBACKEND='TkAgg' if group else 'Agg')
  if not group:
    for name in ('CRAPPY_TEST_GUI', 'DISPLAY', 'WAYLAND_DISPLAY'):
      environment.pop(name, None)
  for phase, timeout in (('--preflight', 45), ('--worker', args.timeout)):
    print(f'{args.directory}: {phase[2:]} (deadline {timeout:g}s)', flush=True)
    try:
      result = run_python(['-m', 'tests.run', args.directory, phase],
                          timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as error:
      print(error.stdout or '', end='', flush=True)
      print(error.stderr or '', end='', file=sys.stderr, flush=True)
      print(f'{args.directory}: {phase[2:]} exceeded {timeout:g}s; '
            'stopped the owned process tree', file=sys.stderr)
      return 1
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode:
      return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())

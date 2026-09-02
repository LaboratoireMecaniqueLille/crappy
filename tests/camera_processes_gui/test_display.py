# coding: utf-8

import os
from pathlib import Path
from platform import system
import subprocess
import sys
from time import sleep
import unittest

import numpy as np

import crappy.blocks.camera_processes.display as display_module
from crappy.blocks.camera_processes.display import Displayer

from tests.camera_process.camera_process_test_base import CameraProcessTestBase


class RecordingOverlay:
  """Overlay object recording and modifying displayed images."""

  def __init__(self, marker: str = '') -> None:
    """Initializes recording state."""

    self.marker = marker
    self.called = False
    self.dtype = None
    self.max_value = None

  def draw(self, img: np.ndarray) -> None:
    """Records the received image and draws a visible marker."""

    self.dtype = img.dtype
    self.max_value = int(np.max(img))
    img[0, 0] = 255
    self.called = True


class TestDisplayer(CameraProcessTestBase):
  """GUI tests for the Displayer CameraProcess."""

  def tearDown(self) -> None:
    """Closes GUI resources left open by tests."""

    try:
      display_module.plt.close('all')
    except RuntimeError:
      pass

    super().tearDown()

  def test_arguments_are_validated(self) -> None:
    """Checks constructor validation."""

    with self.assertRaises(ValueError):
      Displayer(title='invalid', framerate=0, backend='cv2')

    with self.assertRaises(ValueError):
      Displayer(title='invalid', framerate=-1, backend='cv2')

    with self.assertRaises(ValueError):
      Displayer(title='invalid', framerate=1, backend='bad')

  def test_get_data_respects_display_framerate(self) -> None:
    """Checks frame throttling and copied image data."""

    displayer = Displayer(title='Crappy test data', framerate=2,
                          backend='mpl')
    self._process = displayer
    shared = self.make_shared(process=displayer,
                              shape=(2, 3),
                              dtype=np.uint16)

    img0 = np.arange(6, dtype=np.uint16).reshape(2, 3)
    self.write_image(shared, img0, {'ImageUniqueID': 1, 't(s)': 0.1})

    # Fresh Displayer instances should initially throttle according to
    # framerate.
    displayer._last_upd = display_module.time()
    self.assertFalse(displayer._get_data())

    displayer._last_upd = 0
    self.assertTrue(displayer._get_data())
    np.testing.assert_array_equal(displayer.img, img0)
    self.assertEqual(displayer.metadata['ImageUniqueID'], 1)

    img1 = img0 + 10
    self.write_image(shared, img1, {'ImageUniqueID': 2, 't(s)': 0.2})
    displayer._last_upd = display_module.time()
    self.assertFalse(displayer._get_data())
    np.testing.assert_array_equal(displayer.img, img0)

    displayer._last_upd = 0
    self.assertTrue(displayer._get_data())
    np.testing.assert_array_equal(displayer.img, img1)

  def test_cv2_backend_opens_updates_and_closes_real_window(self) -> None:
    """Checks the real OpenCV display lifecycle."""

    recv_conn, _ = self.make_pipe()
    displayer = Displayer(title='Crappy cv2 Displayer test',
                          framerate=100,
                          backend='cv2')
    self._process = displayer
    self.set_test_logger(displayer)
    self.make_shared(process=displayer,
                     shape=(540, 720),
                     dtype=np.uint16,
                     to_draw_conn=recv_conn)
    overlay = RecordingOverlay()

    try:
      displayer.init()
      displayer._overlay = [overlay]

      displayer.img = np.linspace(0, 1023, 540 * 720,
                                  dtype=np.uint16).reshape(540, 720)
      displayer.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
      displayer.loop()

      self.assertTrue(overlay.called)
      self.assertEqual(overlay.dtype, np.uint8)
      self.assertEqual(overlay.max_value, 255)

    finally:
      displayer.finish()

  def test_mpl_backend_opens_updates_and_closes_real_window(self) -> None:
    """Checks the real Matplotlib display lifecycle."""

    # OpenCV and Tk both install process-global NSApplication state on macOS.
    # In production each Displayer has its own process, so exercise TkAgg in a
    # clean interpreter instead of after the OpenCV lifecycle test.
    if (system() == 'Darwin' and
        os.environ.get('CRAPPY_TEST_DISPLAY_BACKEND') != 'mpl'):
      env = os.environ.copy()
      env['CRAPPY_TEST_DISPLAY_BACKEND'] = 'mpl'
      project_root = Path(__file__).resolve().parents[2]
      result = subprocess.run(
        [sys.executable, '-m', 'unittest', '-v',
         f'{__name__}.{type(self).__name__}.{self._testMethodName}'],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15)
      self.assertEqual(result.returncode, 0,
                       f'----- stdout -----\n{result.stdout}\n'
                       f'----- stderr -----\n{result.stderr}')
      return

    # Earlier unit tests may deliberately select Agg. This real-window test
    # owns its graphical precondition instead of depending on suite order.
    backend = display_module.plt.get_backend().lower()
    if (backend in {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'} or
        'inline' in backend):
      try:
        display_module.plt.switch_backend('TkAgg')
      except ImportError as exc:
        self.skipTest(f'No usable Matplotlib GUI backend: {exc}')
      backend = display_module.plt.get_backend().lower()

    self.assertNotIn(backend, {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg',
                               'template'})
    self.assertNotIn('inline', backend)

    recv_conn, _ = self.make_pipe()
    displayer = Displayer(title='Crappy mpl Displayer test',
                          framerate=100,
                          backend='mpl')
    self._process = displayer
    self.set_test_logger(displayer)
    self.make_shared(process=displayer,
                     shape=(540, 720),
                     dtype=np.uint16,
                     to_draw_conn=recv_conn)
    overlay = RecordingOverlay()

    try:
      displayer.init()
      self.assertIsNotNone(displayer._fig)
      self.assertIsNotNone(displayer._ax)

      displayer._overlay = [overlay]

      displayer.img = np.linspace(0, 1023, 540 * 720,
                                  dtype=np.uint16).reshape(540, 720)
      displayer.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
      displayer.loop()

      self.assertTrue(overlay.called)
      self.assertEqual(overlay.dtype, np.uint8)
      self.assertEqual(overlay.max_value, 255)
      self.assertEqual(len(displayer._ax.images), 1)

    finally:
      displayer.finish()

    self.assertNotIn(displayer._fig.number, display_module.plt.get_fignums())

  def test_thread_target_keeps_latest_overlay_message(self) -> None:
    """Checks overlay pipe draining in the Displayer helper thread."""

    recv_conn, send_conn = self.make_pipe()
    displayer = Displayer(title='Crappy overlay Displayer test',
                          framerate=100,
                          backend='mpl')
    self._process = displayer
    self.set_test_logger(displayer)
    self.make_shared(process=displayer,
                     shape=(2, 2),
                     dtype=np.uint8,
                     to_draw_conn=recv_conn)

    overlays = [[RecordingOverlay('first')], [RecordingOverlay('second')]]

    try:
      displayer._overlay_thread = display_module.Thread(
        target=displayer._thread_target)
      displayer._overlay_thread.start()

      send_conn.send(overlays[0])
      send_conn.send(overlays[1])
      for _ in range(100):
        if displayer._overlay and displayer._overlay[0].marker == 'second':
          break
        sleep(0.01)

      self.assertEqual(len(displayer._overlay), 1)
      self.assertEqual(displayer._overlay[0].marker, 'second')

    finally:
      displayer._stop_thread = True
      displayer._stop_event.set()
      if displayer._overlay_thread is not None:
        displayer._overlay_thread.join(1.0)


if __name__ == '__main__':
  unittest.main()

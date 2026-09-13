# coding: utf-8

import logging
import os
from pathlib import Path
from platform import system
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

import numpy as np

import crappy.blocks.vision.display as display_module
from crappy._global import OptionalModule
from crappy.blocks.vision import ImageDisplayer
from crappy.tool.camera_config import Overlay

from tests.vision.vision_test_base import VisionTestBase


class RecordingOverlay(Overlay):
  """Overlay recording the converted image and drawing a visible marker."""

  def __init__(self, marker: str = '') -> None:
    """Initializes call state and an optional test marker."""

    super().__init__()
    self.marker = marker
    self.calls = 0
    self.dtype = None
    self.max_value = None

  def draw(self, image: np.ndarray) -> None:
    """Records the image state and changes its first pixel."""

    self.calls += 1
    self.dtype = image.dtype
    self.max_value = int(np.max(image))
    image.flat[0] = 255


class TestImageDisplayer(VisionTestBase):
  """GUI-designated tests for the ImageDisplayer VisionBlock."""

  def setUp(self) -> None:
    """Resets automatic titles and common VisionBlock state."""

    super().setUp()
    ImageDisplayer._count = 0

  def tearDown(self) -> None:
    """Closes Matplotlib figures that may survive a failed assertion."""

    try:
      display_module.plt.close('all')
    except (RuntimeError, AttributeError):
      pass
    ImageDisplayer._count = 0
    super().tearDown()

  def make_displayer(self, **kwargs) -> ImageDisplayer:
    """Creates and tracks a deterministic explicit-backend displayer."""

    options = {'title': 'Vision display test',
               'backend': 'cv2',
               'framerate': 5}
    options.update(kwargs)
    displayer = ImageDisplayer(**options)
    self.track_block(displayer)
    return displayer

  @staticmethod
  def add_image_input(displayer: ImageDisplayer,
                      name: str = 'display-image') -> Mock:
    """Registers a minimal input ImageLink double."""

    link = Mock()
    link.name = name
    displayer.add_img_input(link)
    return link

  def feed_image(self,
                 displayer: ImageDisplayer,
                 image: np.ndarray,
                 metadata=None,
                 name: str = 'display-image') -> None:
    """Installs one received image and makes receive_imgs report it."""

    if name not in displayer.last_received:
      self.add_image_input(displayer, name)
    if metadata is None:
      metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    displayer.last_received[name].metadata = metadata
    displayer.last_received[name].img = image
    displayer.receive_imgs = Mock(return_value=[name])

  def test_constructor_sets_titles_backend_and_rate(self) -> None:
    """Checks automatic title uniqueness and explicit display settings."""

    first = ImageDisplayer(backend='cv2')
    second = ImageDisplayer(backend='mpl', framerate=2, freq=None)
    self.track_block(first)
    self.track_block(second)

    self.assertEqual(first._title, 'Displayer 1')
    self.assertEqual(second._title, 'Displayer 2')
    self.assertEqual(first._backend, 'cv2')
    self.assertEqual(second._backend, 'mpl')
    self.assertEqual(second._framerate, 2)

  def test_constructor_validates_title_rate_and_backend(self) -> None:
    """Checks user-facing display option validation."""

    invalid = (
      {'title': ''},
      {'title': 1},
      {'framerate': 0},
      {'framerate': -1},
      {'framerate': '5'},
      {'framerate': 6, 'freq': 5},
      {'backend': 'invalid'},
    )
    for options in invalid:
      with self.subTest(options=options):
        defaults = {'backend': 'cv2'}
        defaults.update(options)
        with self.assertRaises(ValueError):
          ImageDisplayer(**defaults)

  def test_constructor_auto_selects_backend_and_reports_no_backend(self
                                                                  ) -> None:
    """Checks OpenCV preference, Matplotlib fallback, and missing dependencies."""

    available_cv2 = Mock()
    with patch.object(display_module, 'cv2', available_cv2):
      displayer = ImageDisplayer(backend=None)
    self.track_block(displayer)
    self.assertEqual(displayer._backend, 'cv2')

    missing_cv2 = OptionalModule('missing-cv2')
    available_plt = Mock()
    available_plt.Figure = object()
    with (patch.object(display_module, 'cv2', missing_cv2),
          patch.object(display_module, 'plt', available_plt)):
      displayer = ImageDisplayer(backend=None)
    self.track_block(displayer)
    self.assertEqual(displayer._backend, 'mpl')

    with (patch.object(display_module, 'cv2', missing_cv2),
          patch.object(display_module, 'plt',
                       OptionalModule('missing-matplotlib'))):
      with self.assertRaises(ModuleNotFoundError):
        ImageDisplayer(backend=None)

  def test_prepare_validates_supported_topology(self) -> None:
    """Checks the exactly-one-input and no-output ImageLink topology."""

    displayer = self.make_displayer()
    with self.assertRaises(IOError):
      displayer.prepare()

    self.add_image_input(displayer)
    displayer.img_outputs.append(Mock())
    with self.assertRaises(IOError):
      displayer.prepare()

    displayer.img_outputs.clear()
    self.add_image_input(displayer, 'second-image')
    with self.assertRaises(IOError):
      displayer.prepare()

  def test_prepare_and_finish_dispatch_cv2_backend(self) -> None:
    """Checks the OpenCV window and inherited buffer lifecycle dispatch."""

    displayer = self.make_displayer()
    self.add_image_input(displayer)
    displayer._prepare_cv2 = Mock()
    displayer._finish_cv2 = Mock()

    with patch.object(display_module.VisionBlock, 'prepare') as inherited:
      displayer.prepare()
    displayer._prepare_cv2.assert_called_once_with()
    inherited.assert_called_once_with()

    with patch.object(display_module.VisionBlock, 'finish') as inherited:
      displayer.finish()
    displayer._finish_cv2.assert_called_once_with()
    inherited.assert_called_once_with()

  def test_prepare_and_finish_dispatch_mpl_backend(self) -> None:
    """Checks the Matplotlib window and inherited buffer lifecycle dispatch."""

    displayer = self.make_displayer(backend='mpl')
    self.add_image_input(displayer)
    displayer._prepare_mpl = Mock()
    displayer._finish_mpl = Mock()

    with patch.object(display_module.VisionBlock, 'prepare') as inherited:
      displayer.prepare()
    displayer._prepare_mpl.assert_called_once_with()
    inherited.assert_called_once_with()

    with patch.object(display_module.VisionBlock, 'finish') as inherited:
      displayer.finish()
    displayer._finish_mpl.assert_called_once_with()
    inherited.assert_called_once_with()

  def test_loop_consumes_overlays_before_rate_limit(self) -> None:
    """Checks overlay state remains current even when display is throttled."""

    displayer = self.make_displayer(framerate=2, display_freq=True)
    overlay = RecordingOverlay('new')
    regular = Mock()
    regular.name = 'overlay-link'
    regular.recv_last.return_value = {'overlay': [overlay, None]}
    displayer.inputs = [regular]
    displayer._last_upd = 10.0
    displayer.receive_imgs = Mock()
    displayer._print_freq = Mock()

    with patch.object(display_module, 'time', return_value=10.1):
      displayer.loop()

    self.assertEqual(displayer._overlay_buffer[regular.name], (overlay, None))
    displayer.receive_imgs.assert_not_called()
    displayer._print_freq.assert_called_once_with(img_handled=False)

  def test_loop_retains_and_clears_each_links_overlays(self) -> None:
    """Checks latest valid overlays persist and empty iterables clear them."""

    displayer = self.make_displayer()
    overlay = RecordingOverlay()
    regular = Mock()
    regular.name = 'overlay-link'
    regular.recv_last.side_effect = [
      {'overlay': [overlay]},
      {},
      {'overlay': []},
    ]
    displayer.inputs = [regular]
    displayer.receive_imgs = Mock(return_value=[])

    for timestamp in (10.0, 11.0, 12.0):
      displayer._last_upd = float('-inf')
      with patch.object(display_module, 'time', return_value=timestamp):
        displayer.loop()

      if timestamp < 12:
        self.assertEqual(displayer._overlay_buffer[regular.name], (overlay,))

    self.assertEqual(displayer._overlay_buffer[regular.name], tuple())

  def test_loop_ignores_malformed_overlays_and_throttles_warnings(self) -> None:
    """Checks bad overlay iterables do not replace prior valid state."""

    displayer = self.make_displayer()
    retained = RecordingOverlay()
    regular = Mock()
    regular.name = 'overlay-link'
    regular.recv_last.side_effect = [
      {'overlay': 1},
      {'overlay': [object()]},
    ]
    displayer.inputs = [regular]
    displayer._overlay_buffer[regular.name] = (retained,)
    displayer._last_upd = 10.0
    displayer.log = Mock()

    for timestamp in (10.1, 11.0):
      with patch.object(display_module, 'time', return_value=timestamp):
        displayer.loop()

    self.assertEqual(displayer._overlay_buffer[regular.name], (retained,))
    warnings = [args for args, _ in displayer.log.call_args_list
                if args[0] == logging.WARNING]
    self.assertEqual(len(warnings), 1)

  def test_loop_casts_draws_updates_and_sends_metadata(self) -> None:
    """Checks image conversion, overlay drawing, backend update, and output."""

    displayer = self.make_displayer(display_freq=True)
    overlay = RecordingOverlay()
    regular = Mock()
    regular.name = 'overlay-link'
    regular.recv_last.return_value = {'overlay': [overlay]}
    displayer.inputs = [regular]
    displayer._update_cv2 = Mock()
    displayer.send = Mock()
    displayer._print_freq = Mock()
    image = np.linspace(0, 1023, 12, dtype=np.uint16).reshape(3, 4)
    original = image.copy()
    metadata = {'ImageUniqueID': 7, 't(s)': 1.25, 'camera': 'fake'}
    self.feed_image(displayer, image, metadata)

    with patch.object(display_module, 'time', return_value=10.0):
      displayer.loop()

    self.assertEqual(overlay.calls, 1)
    self.assertEqual(overlay.dtype, np.uint8)
    self.assertEqual(overlay.max_value, 255)
    displayed = displayer._update_cv2.call_args.args[0]
    self.assertEqual(displayed.dtype, np.uint8)
    self.assertEqual(displayed.flat[0], 255)
    np.testing.assert_array_equal(image, original)
    displayer.send.assert_called_once_with({
      't(s)': 1.25,
      'img_index': 7,
      'meta': metadata,
    })
    displayer._print_freq.assert_called_once_with(img_handled=True)

  def test_loop_copies_uint8_image_before_drawing_overlay(self) -> None:
    """Checks overlays never mutate the consumer's retained source frame."""

    displayer = self.make_displayer()
    overlay = RecordingOverlay()
    displayer._overlay_buffer['manual'] = (overlay,)
    displayer._update_cv2 = Mock()
    image = np.zeros((2, 3), dtype=np.uint8)
    self.feed_image(displayer, image)

    with patch.object(display_module, 'time', return_value=10.0):
      displayer.loop()

    self.assertEqual(image.flat[0], 0)
    self.assertEqual(displayer._update_cv2.call_args.args[0].flat[0], 255)

  def test_loop_skips_when_no_new_image(self) -> None:
    """Checks idle polling and handled-image frequency accounting."""

    displayer = self.make_displayer(display_freq=True)
    displayer.receive_imgs = Mock(return_value=[])
    displayer._print_freq = Mock()

    with patch.object(display_module, 'time', return_value=10.0):
      displayer.loop()

    displayer._print_freq.assert_called_once_with(img_handled=False)
    self.assertEqual(displayer._last_upd, 10.0)

  def test_loop_requires_metadata_and_mandatory_keys(self) -> None:
    """Checks clear failures for incomplete received image metadata."""

    displayer = self.make_displayer()
    displayer._update_cv2 = Mock()
    image = np.zeros((2, 2), dtype=np.uint8)
    self.feed_image(displayer, image)

    displayer.last_received['display-image'].metadata = None
    with patch.object(display_module, 'time', return_value=10.0):
      with self.assertRaises(RuntimeError):
        displayer.loop()

    for metadata in ({'t(s)': 0.1}, {'ImageUniqueID': 1}):
      with self.subTest(metadata=metadata):
        displayer._last_upd = float('-inf')
        displayer.last_received['display-image'].metadata = metadata
        with patch.object(display_module, 'time', return_value=11.0):
          with self.assertRaises(RuntimeError):
            displayer.loop()

  def test_update_cv2_downscales_large_images(self) -> None:
    """Checks OpenCV resize bounds, display, and event pumping."""

    fake_cv2 = Mock()
    fake_cv2.resize.return_value = resized = np.zeros((360, 640),
                                                      dtype=np.uint8)
    displayer = self.make_displayer()
    image = np.zeros((720, 1280), dtype=np.uint8)

    with patch.object(display_module, 'cv2', fake_cv2):
      displayer._update_cv2(image)

    fake_cv2.resize.assert_called_once_with(image, (640, 360))
    fake_cv2.imshow.assert_called_once_with(displayer._title, resized)
    fake_cv2.waitKey.assert_called_once_with(1)

  def test_update_mpl_subsamples_large_images(self) -> None:
    """Checks Matplotlib integer subsampling and redraw calls."""

    fake_plt = Mock()
    displayer = self.make_displayer(backend='mpl')
    displayer._ax = Mock()
    image = np.zeros((960, 1280), dtype=np.uint8)

    with patch.object(display_module, 'plt', fake_plt):
      displayer._update_mpl(image)

    displayer._ax.clear.assert_called_once_with()
    displayed = displayer._ax.imshow.call_args.args[0]
    self.assertEqual(displayed.shape, (480, 640))
    self.assertEqual(displayer._ax.imshow.call_args.kwargs, {'cmap': 'gray'})
    fake_plt.pause.assert_called_once_with(0.001)
    fake_plt.show.assert_called_once_with()

  def _run_real_backend(self, backend: str) -> None:
    """Exercises a real window lifecycle in an isolated interpreter."""

    if system() == 'Linux' and not os.environ.get('DISPLAY'):
      self.skipTest('No display is available for GUI lifecycle testing')

    if backend == 'cv2' and isinstance(display_module.cv2, OptionalModule):
      self.skipTest('OpenCV is not available')

    code = textwrap.dedent(f'''\
      import numpy as np
      import crappy.blocks.vision.display as module
      from crappy.blocks.vision import ImageDisplayer

      if {backend!r} == 'mpl':
        current = module.plt.get_backend().lower()
        non_gui = {{'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'}}
        if current in non_gui or 'inline' in current:
          module.plt.switch_backend('TkAgg')

      displayer = ImageDisplayer(title='Crappy vision {backend} test',
                                 backend={backend!r}, framerate=100)
      image = np.zeros((540, 720), dtype=np.uint8)
      try:
        if {backend!r} == 'cv2':
          displayer._prepare_cv2()
          displayer._update_cv2(image)
        else:
          displayer._prepare_mpl()
          assert displayer._fig is not None
          assert displayer._ax is not None
          displayer._update_mpl(image)
      finally:
        if {backend!r} == 'cv2':
          displayer._finish_cv2()
        else:
          figure = displayer._fig
          displayer._finish_mpl()
          assert figure.number not in module.plt.get_fignums()
    ''')
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=Path(__file__).resolve().parents[2],
                            env=os.environ.copy(),
                            capture_output=True,
                            text=True,
                            timeout=20)
    self.assertEqual(result.returncode, 0,
                     f'----- stdout -----\n{result.stdout}\n'
                     f'----- stderr -----\n{result.stderr}')

  def test_cv2_backend_real_window_lifecycle(self) -> None:
    """Checks a real OpenCV window opens, updates, and closes."""

    self._run_real_backend('cv2')

  def test_mpl_backend_real_window_lifecycle(self) -> None:
    """Checks a real Matplotlib window opens, updates, and closes."""

    self._run_real_backend('mpl')

if __name__ == '__main__':
  unittest.main()

# coding: utf-8

from unittest.mock import Mock, patch

import crappy.blocks.vision.display as display_module
from crappy.blocks.vision import ImageDisplayer

from .vision_test_base import VisionTestBase


class TestImageDisplayerCleanup(VisionTestBase):
  """Checks window ownership without opening GUI windows."""

  def make_displayer(self, backend: str) -> ImageDisplayer:
    """Creates and retains a displayer using a mocked backend."""

    displayer = ImageDisplayer(backend=backend)
    self.track_block(displayer)
    return displayer

  def test_finish_without_prepare_does_not_close_window(self) -> None:
    """An unopened window is not a resource owned by this Block."""

    for backend, module in (('cv2', 'cv2'), ('mpl', 'plt')):
      with self.subTest(backend=backend):
        displayer = self.make_displayer(backend)
        with patch.object(display_module, module) as library:
          displayer.finish()
        library.destroyWindow.assert_not_called()
        library.close.assert_not_called()

  def test_failed_buffer_prepare_still_closes_opened_window(self) -> None:
    """Window acquisition precedes shared-memory attachment."""

    for backend, module in (('cv2', 'cv2'), ('mpl', 'plt')):
      with self.subTest(backend=backend):
        displayer = self.make_displayer(backend)
        displayer.img_inputs.append(Mock())
        with (patch.object(display_module, module) as library,
              patch.object(display_module.VisionBlock, 'prepare',
                           side_effect=RuntimeError('buffer attach'))):
          library.WINDOW_NORMAL = 1
          library.WINDOW_KEEPRATIO = 2
          figure, axis = Mock(), Mock()
          library.subplots.return_value = (figure, axis)
          with self.assertRaises(RuntimeError):
            displayer.prepare()
          self.assertTrue(displayer._window_opened)
          displayer.finish()
          displayer.finish()
          if backend == 'cv2':
            library.destroyWindow.assert_called_once_with(displayer._title)
          else:
            library.close.assert_called_once_with(figure)
        self.assertFalse(displayer._window_opened)

  def test_failed_window_prepare_does_not_close_unopened_window(self) -> None:
    """A failed backend creation does not claim ownership of other windows."""

    for backend, module in (('cv2', 'cv2'), ('mpl', 'plt')):
      with self.subTest(backend=backend):
        displayer = self.make_displayer(backend)
        displayer.img_inputs.append(Mock())
        with patch.object(display_module, module) as library:
          library.WINDOW_NORMAL = 1
          library.WINDOW_KEEPRATIO = 2
          library.namedWindow.side_effect = RuntimeError('window creation')
          library.subplots.side_effect = RuntimeError('window creation')
          with self.assertRaises(RuntimeError):
            displayer.prepare()
          displayer.finish()
        library.destroyWindow.assert_not_called()
        library.close.assert_not_called()

  def test_finish_attempts_memory_and_window_cleanup_and_retries(self) -> None:
    """Both failures are reported, and a failed window close can be retried."""

    displayer = self.make_displayer('cv2')
    displayer._window_opened = True
    errors = [RuntimeError('buffer close'), RuntimeError('window close')]
    with (patch.object(display_module.VisionBlock, 'finish',
                       side_effect=errors[0]) as inherited,
          patch.object(displayer, '_finish_cv2',
                       side_effect=errors[1]) as close):
      with self.assertRaises(ExceptionGroup) as caught:
        displayer.finish()
      self.assertEqual(caught.exception.exceptions, tuple(errors))
      inherited.assert_called_once_with()
      close.assert_called_once_with()
      self.assertTrue(displayer._window_opened)
      close.side_effect = None
      inherited.side_effect = None
      displayer.finish()
      displayer.finish()
      self.assertEqual(close.call_count, 2)
    self.assertFalse(displayer._window_opened)

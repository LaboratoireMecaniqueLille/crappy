# coding: utf-8

"""Qt lifetime, cleanup, cancellation, and callback failures."""

from unittest.mock import Mock, patch
from multiprocessing import Event, Queue

from ._fixtures import PyQtConfigTestCase


class TestLifecycle(PyQtConfigTestCase):
  def test_stop_before_start_is_idempotent(self) -> None:
    """A window can release its queues before starting any worker or timers."""

    config = self.make_config()
    config.stop()
    config.stop()

    config._histogram_process.start.assert_not_called()
    config._histogram_process.join.assert_not_called()
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)
    self.assertFalse(config._acquisition_timer.isActive())
    self.assertFalse(config._indicator_timer.isActive())
    self.assertFalse(config._shutdown_timer.isActive())

  def test_shutdown_closes_without_validating_selection(self) -> None:
    """A Block shutdown cancels an incomplete selection without validation."""

    from PyQt6.QtCore import QTimer
    from crappy.tool.camera_config import Box
    from crappy.tool.camera_config.pyqt import PyQtDISCorrelConfig

    config = self.make_config(PyQtDISCorrelConfig, Box())
    stop_event = Event()
    config.watch_shutdown(stop_event.is_set)
    QTimer.singleShot(0, stop_event.set)
    with (patch.object(config, '_validate_close') as validate,
          patch.object(config, '_on_valid_close') as finalize):
      config.run()

    validate.assert_not_called()
    finalize.assert_not_called()
    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)

  def test_shutdown_before_run_does_not_start_histogram(self) -> None:
    """An already canceled Block never opens a window or starts its worker."""

    config = self.make_config()
    config.watch_shutdown(lambda: True)
    config.run()

    config._histogram_process.start.assert_not_called()
    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)

  def test_start_failure_cleans_unstarted_process(self) -> None:
    """A failed Qt histogram startup still closes timers, queues, and window."""

    config = self.make_config()
    config._histogram_process.start.side_effect = RuntimeError('cannot start')
    with self.assertRaisesRegex(RuntimeError, 'cannot start'):
      config.run()

    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)

  def test_constructor_failure_cleans_resources(self) -> None:
    """A failed Qt layout releases resources created before the GUI controls."""

    from crappy.tool.camera_config.pyqt import PyQtCameraConfig

    class BrokenConfig(PyQtCameraConfig):
      instance = None

      def _set_layout(self) -> None:
        type(self).instance = self
        raise ValueError('layout failed')

    with self.assertRaisesRegex(ValueError, 'layout failed'):
      BrokenConfig(self.camera, self.log_queue, None, 30, None)

    config = BrokenConfig.instance
    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)
    self.assertFalse(config._histogram_process.is_alive())

  def test_histogram_constructor_failure_closes_created_queues(self) -> None:
    """Qt cleans up partially created resources before lifecycle setup."""

    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    queues = []

    def make_queue(*args, **kwargs):
      queue = Queue(*args, **kwargs)
      queues.append(queue)
      return queue

    with (patch.object(pyqt_config, 'Queue', side_effect=make_queue),
          patch.object(pyqt_config, 'HistogramProcess',
                       side_effect=RuntimeError('histogram failed'))):
      with self.assertRaisesRegex(RuntimeError, 'histogram failed'):
        pyqt_config.PyQtCameraConfig(self.camera, self.log_queue, None, 30, None)

    self.assertEqual(len(queues), 2)
    self.assertTrue(all(queue._closed for queue in queues))

  def test_partial_core_failure_keeps_original_exception(self) -> None:
    """A failure before lifecycle creation does not trigger close validation."""

    from crappy.tool.camera_config.pyqt import PyQtCameraConfig

    class FailingConfig(PyQtCameraConfig):
      def _create_local_settings(self):
        raise ValueError('local setting')

    with self.assertRaisesRegex(ValueError, 'local setting'):
      FailingConfig(self.camera, self.log_queue, None, 30, None)

  def test_callback_error_reaches_run(self) -> None:
    """Qt callback errors escape run() after the window and process close."""

    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    config = self.make_config()
    config._update_img = Mock(side_effect=ValueError('frame'))
    with patch.object(pyqt_config.QMessageBox, 'critical'):
      with self.assertRaisesRegex(ValueError, 'frame'):
        config.run()
    self.assertTrue(config._lifecycle.closed)
    self.assertTrue(config._window_closed)

  def test_run_closes_normally(self) -> None:
    """run() starts acquisition, waits, and returns after a valid close."""

    from PyQt6.QtCore import QTimer

    config = self.make_config()
    QTimer.singleShot(0, config.finish)
    config.run()

    self.assertTrue(config._lifecycle.closed)
    self.assertTrue(config._window_closed)

  def test_callback_keyboard_interrupt_closes_silently(self) -> None:
    """Qt timer cancellation closes resources without a dialog or error."""

    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    config = self.make_config()
    interrupt = KeyboardInterrupt()
    config._update_img = Mock(side_effect=interrupt)
    with (patch.object(pyqt_config.QMessageBox, 'critical') as dialog,
          patch.object(config._lifecycle, '_log') as log):
      with self.assertRaises(KeyboardInterrupt) as raised:
        config.run()

    self.assertIs(raised.exception, interrupt)
    dialog.assert_not_called()
    log.assert_not_called()
    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)
    self.assertFalse(config._histogram_process.is_alive())

  def test_event_keyboard_interrupts_close_silently(self) -> None:
    """Early signal, mouse/resize, and close interrupts take the same path."""

    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QCloseEvent
    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    for path in ('signal', 'event_filter', 'close'):
      with self.subTest(path=path):
        config = self.make_config()
        interrupt = KeyboardInterrupt()
        with (patch.object(pyqt_config.QMessageBox, 'critical') as dialog,
              patch.object(config._lifecycle, '_log') as log):
          if path == 'signal':
            config._guard(Mock(side_effect=interrupt))()
          elif path == 'event_filter':
            config._read_image_geometry = Mock(side_effect=interrupt)
            config.eventFilter(config._img_canvas, QEvent(QEvent.Type.Resize))
          else:
            config.finish = Mock(side_effect=interrupt)
            event = QCloseEvent()
            config.closeEvent(event)
            self.assertTrue(event.isAccepted())
          with self.assertRaises(KeyboardInterrupt) as raised:
            config.run()

        self.assertIs(raised.exception, interrupt)
        dialog.assert_not_called()
        log.assert_not_called()
        self.assertTrue(config._window_closed)
        self.assertTrue(config._img_in._closed)
        self.assertTrue(config._img_out._closed)

  def test_run_keyboard_interrupt_closes_silently(self) -> None:
    """An interrupt outside a Qt callback also cleans up before propagating."""

    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    config = self.make_config()
    with (patch.object(config, 'start', side_effect=KeyboardInterrupt),
          patch.object(pyqt_config.QMessageBox, 'critical') as dialog,
          patch.object(config._lifecycle, '_log') as log):
      with self.assertRaises(KeyboardInterrupt):
        config.run()

    dialog.assert_not_called()
    log.assert_not_called()
    self.assertTrue(config._window_closed)
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)

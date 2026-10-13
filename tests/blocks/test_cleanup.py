# coding: utf-8

"""Headless fault-injection checks for software-owned shutdown resources."""

from threading import Thread
from multiprocessing import Event
from unittest.mock import Mock, patch

import crappy.blocks.grapher as grapher_module
from crappy.blocks.camera_processes.display import Displayer
from crappy.blocks.client_server import ClientServer
from crappy.blocks.grapher import Grapher
from crappy.blocks.ucontroller import UController
from crappy._failure import FailureManager
from crappy.links import Link
from crappy.links import link_graph

from .test_gui_backends import GUI_BLOCKS, GUIBlockTestBase
from tests.block.block_test_base import exception_leaves
from ..block import TestBlock


class TestSoftwareCleanup(GUIBlockTestBase):
  """Checks independent cleanup steps, error identity, and safe retries."""

  def test_gui_failures_do_not_skip_other_windows_or_events(self) -> None:
    """All GUI resources are attempted before several errors are raised."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = block_type(*args)
        tk_window, qt_window, app = Mock(), Mock(), Mock()
        errors = (RuntimeError('Tk close'), OSError('Qt close'),
                  ValueError('Qt events'))
        tk_window.destroy.side_effect = errors[0]
        qt_window.close.side_effect = errors[1]
        app.processEvents.side_effect = errors[2]
        if block_type.__name__ == 'Dashboard':
          block._dashboard = tk_window
        else:
          block._root = tk_window
        block._qt_window, block._qt_app = qt_window, app

        with self.assertRaises(ExceptionGroup) as caught:
          block.finish()
        self.assertEqual(caught.exception.exceptions, errors)
        for error in errors:
          self.assertIn(block_type.__name__, error.__notes__[0])
        tk_window.destroy.side_effect = None
        qt_window.close.side_effect = None
        app.processEvents.side_effect = None
        block.finish()
        block.finish()
        self.assertEqual(tk_window.destroy.call_count, 2)
        self.assertEqual(qt_window.close.call_count, 2)
        self.assertEqual(app.processEvents.call_count, 2)
        app.quit.assert_not_called()

  def test_gui_interrupt_retains_its_cause_and_other_failures(self) -> None:
    """Interrupts remain group members without changing their causes."""

    block_type, _, args = GUI_BLOCKS[0]
    block = block_type(*args)
    window, app = Mock(), Mock()
    interrupt, error = KeyboardInterrupt(), OSError('events')
    cause = RuntimeError('earlier failure')
    interrupt.__cause__ = cause
    window.close.side_effect = interrupt
    app.processEvents.side_effect = error
    block._qt_window, block._qt_app = window, app
    with self.assertRaises(BaseExceptionGroup) as caught:
      block.finish()
    self.assertIn(interrupt, exception_leaves(caught.exception))
    self.assertIs(interrupt.__cause__, cause)
    self.assertEqual(exception_leaves(caught.exception), (interrupt, error))

  def test_gui_refused_close_retains_window_and_reprocesses_events(self
                                                                 ) -> None:
    """A failed close is retried without losing its deferred Qt events."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = block_type(*args)
        window, app = Mock(), Mock()
        window.close.return_value = False
        block._qt_window, block._qt_app = window, app
        with self.assertRaises(ExceptionGroup) as caught:
          block.finish()
        self.assertIsInstance(exception_leaves(caught.exception)[0],
                              RuntimeError)
        self.assertRegex(str(exception_leaves(caught.exception)[0]),
                         'refused to close')
        self.assertIs(block._qt_window, window)
        app.processEvents.assert_called_once_with()
        window.close.return_value = True
        block.finish()
        block.finish()
        self.assertIsNone(block._qt_window)
        self.assertEqual(window.close.call_count, 2)
        self.assertEqual(app.processEvents.call_count, 2)

  def test_grapher_closes_matplotlib_after_qt_failure(self) -> None:
    """A Qt close cannot skip event processing or Matplotlib cleanup."""

    with patch.object(grapher_module, 'warn'):
      block = Grapher(('x', 'y'), plotter='mpl', backend='TkAgg')
    window, app, figure = Mock(), Mock(), Mock()
    block._qt_plot, block._qt_app, block._figure = window, app, figure
    error = RuntimeError('Qt close')
    window.close.side_effect = error
    with patch.object(grapher_module, 'plt') as plt:
      with self.assertRaises(ExceptionGroup) as caught:
        block.finish()
      self.assertEqual(exception_leaves(caught.exception), (error,))
      plt.close.assert_called_once_with(figure)
      app.processEvents.assert_called_once_with()
      window.close.side_effect = None
      block.finish()
      block.finish()
      plt.close.assert_called_once_with(figure)

  def test_displayer_pipe_close_during_shutdown_ends_receiver(self) -> None:
    """Closing the overlay endpoint cannot leave a noisy receiver thread."""

    process = Displayer('cleanup', 10, backend='cv2')
    process._stop_event = Event()
    process._to_draw_conn = Mock()

    def close_during_receive() -> None:
      process._stop_thread = True
      raise EOFError

    process._to_draw_conn.poll.return_value = True
    process._to_draw_conn.recv.side_effect = close_during_receive
    process._thread_target()
    self.assertTrue(process._stop_thread)

  def test_client_failures_do_not_skip_broker_or_stdout(self) -> None:
    """MQTT errors cannot prevent broker reaping and stdout cleanup."""

    block = ClientServer(broker=True)
    client, proc, stream, reader = Mock(), Mock(), Mock(), Mock()
    proc.stdout = stream
    reader.is_alive.return_value = False
    block._client, block._proc, block._reader = client, proc, reader
    block._client_loop_started = block._reader_started = True
    errors = (RuntimeError('disconnect'), OSError('loop stop'),
              ValueError('terminate'), RuntimeError('stdout'))
    client.disconnect.side_effect = errors[0]
    client.loop_stop.side_effect = errors[1]
    proc.terminate.side_effect = errors[2]
    stream.close.side_effect = errors[3]
    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()
    self.assertEqual(caught.exception.exceptions, errors)
    proc.wait.assert_called_once_with(timeout=15)
    reader.join.assert_called_once_with(0.2)
    self.assertTrue(block._stop_mosquitto)
    client.disconnect.side_effect = None
    client.loop_stop.side_effect = None
    stream.close.side_effect = None
    block.finish()
    block.finish()
    proc.terminate.assert_called_once_with()
    self.assertEqual(stream.close.call_count, 2)
    self.assertIsNone(block._proc)
    self.assertIsNone(block._client)

  def test_live_broker_reader_keeps_stdout_until_retry(self) -> None:
    """Closing a stream held by a live reader must not block shutdown."""

    block = ClientServer(broker=True)
    proc, reader = Mock(), Mock()
    block._proc, block._reader = proc, reader
    block._reader_started = True
    reader.is_alive.return_value = True
    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()
    self.assertIsInstance(exception_leaves(caught.exception)[0],
                          RuntimeError)
    self.assertRegex(str(exception_leaves(caught.exception)[0]), 'reader thread')
    proc.stdout.close.assert_not_called()
    reader.is_alive.return_value = False
    block.finish()
    block.finish()
    proc.terminate.assert_called_once_with()
    proc.stdout.close.assert_called_once_with()

  def test_displayer_joins_thread_after_window_failure(self) -> None:
    """A failing display backend cannot skip overlay-thread cleanup."""

    process = Displayer('cleanup', 10, backend='cv2')
    thread = Mock(spec=Thread)
    thread.is_alive.return_value = False
    process._overlay_thread = thread
    process._overlay_started = process._window_opened = True
    window_error, interrupt = OSError('window'), KeyboardInterrupt()
    thread.join.side_effect = interrupt
    with patch.object(process, '_finish_cv2', side_effect=window_error):
      with self.assertRaises(BaseExceptionGroup) as caught:
        process.finish()
    self.assertIn(interrupt, exception_leaves(caught.exception))
    self.assertEqual(exception_leaves(caught.exception),
                     (window_error, interrupt))
    self.assertIsNone(interrupt.__cause__)
    self.assertTrue(process._stop_thread)
    self.assertIsNone(process._overlay_thread)
    with patch.object(process, '_finish_cv2') as close:
      process.finish()
      process.finish()
    close.assert_called_once_with()

  def test_displayer_does_not_join_a_failed_thread_start(self) -> None:
    """A Thread object is not evidence that start() succeeded."""

    process = Displayer('cleanup', 10, backend='cv2')
    error = RuntimeError('thread start')
    thread = Mock(spec=Thread)
    thread.start.side_effect = error
    with patch('crappy.blocks.camera_processes.display.Thread',
               return_value=thread):
      with self.assertRaises(RuntimeError):
        process.init()
    process.finish()
    thread.join.assert_not_called()

  def test_ucontroller_closes_bus_after_stop_write_failure(self) -> None:
    """A protocol error cannot prevent releasing the serial connection."""

    block = UController()
    bus = Mock()
    block._bus = bus
    errors = (RuntimeError('write'), OSError('close'))
    bus.write.side_effect, bus.close.side_effect = errors
    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()
    self.assertEqual(caught.exception.exceptions, errors)
    bus.close.assert_called_once_with()
    bus.write.side_effect = bus.close.side_effect = None
    block.finish()
    block.finish()
    self.assertIsNone(block._bus)
    self.assertEqual(bus.close.call_count, 2)

  def test_link_closes_both_endpoints_and_retries_only_failures(self) -> None:
    """Links release both endpoints without repeating a successful close."""

    source, target = TestBlock(), TestBlock()
    link = Link(source, target)
    link._in.close()
    link._out.close()
    inlet, outlet = Mock(), Mock()
    link._in, link._out = inlet, outlet
    error = OSError('input close')
    inlet.close.side_effect = error
    with self.assertRaises(ExceptionGroup) as caught:
      link.close()
    self.assertEqual(exception_leaves(caught.exception), (error,))
    outlet.close.assert_called_once_with()
    inlet.close.side_effect = None
    link.close()
    link.close()
    self.assertEqual(inlet.close.call_count, 2)
    outlet.close.assert_called_once_with()

  def test_owned_cleanup_reports_without_outer_duplicate(self) -> None:
    """The Block lifecycle and its cleanup share one reporting history."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = block_type(*args)
        manager = block._failure_manager = FailureManager(block.name, 'finish')
        manager._logger = Mock()
        block._logger = manager._logger
        queue = Mock()
        manager.set_queue(queue)
        window, app = Mock(), Mock()
        first, interrupt, last = (OSError('first close'), KeyboardInterrupt(),
                                  ValueError('events'))
        nested = BaseExceptionGroup('original nested group',
                                     [first, interrupt])
        cause = RuntimeError('original cause')
        nested.__cause__ = cause
        nested.add_note('original note')
        window.close.side_effect = nested

        def fail_events() -> None:
          manager._logger.error.assert_called_once()
          queue.put.assert_called_once()
          raise last

        app.processEvents.side_effect = fail_events
        block._qt_window, block._qt_app = window, app
        with self.assertRaises(BaseExceptionGroup) as caught:
          with manager.collect('Block lifecycle'):
            with manager.step('finish'):
              block.finish()
        self.assertEqual(exception_leaves(caught.exception),
                         (first, interrupt, last))
        self.assertIs(nested.__cause__, cause)
        self.assertIn('original note', nested.__notes__)
        self.assertEqual(len(manager.snapshot()), 2)
        self.assertEqual(manager._logger.error.call_count, 2)
        self.assertEqual(queue.put.call_count, 2)
        window.close.side_effect = app.processEvents.side_effect = None
        block.finish()
        block.finish()
        self.assertEqual(len(manager.snapshot()), 2)
        self.assertEqual(manager._logger.error.call_count, 2)

  def test_failed_cleanup_diagnostic_does_not_skip_resources(self) -> None:
    """A failed progress log cannot prevent independent resource releases."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = block_type(*args)
        window, app = Mock(), Mock()
        block._qt_window, block._qt_app = window, app
        error = OSError('progress log')
        with patch.object(block, 'log', side_effect=error):
          with self.assertRaises(ExceptionGroup) as caught:
            block.finish()
        self.assertEqual(exception_leaves(caught.exception), (error,))
        window.close.assert_called_once_with()
        app.processEvents.assert_called_once_with()
        self.assertIsNone(block._qt_window)

  def test_link_rollback_preserves_origin_and_endpoint_failures(self) -> None:
    """Failed registration cannot be replaced by its failed rollback."""

    source, target = TestBlock(), TestBlock()
    origin, inlet_error, interrupt = (ValueError('registration'),
                                      OSError('input close'),
                                      KeyboardInterrupt())
    cause = RuntimeError('original cause')
    origin.__cause__ = cause
    inlet, outlet = Mock(), Mock()
    inlet.close.side_effect, outlet.close.side_effect = inlet_error, interrupt
    manager = FailureManager('link rollback', 'initialization')
    manager._logger = Mock()
    with (patch('crappy.links.link.Pipe', return_value=(inlet, outlet)),
          patch.object(link_graph, 'add_edge', side_effect=origin),
          patch('crappy.links.link.FailureManager', return_value=manager)):
      with self.assertRaises(BaseExceptionGroup) as caught:
        Link(source, target)
    self.assertEqual(exception_leaves(caught.exception),
                     (origin, inlet_error, interrupt))
    self.assertIs(origin.__cause__, cause)
    inlet.close.assert_called_once_with()
    outlet.close.assert_called_once_with()
    self.assertEqual(len(manager.snapshot()), 3)
    self.assertEqual(manager._logger.error.call_count, 3)
    self.assertEqual(source.outputs, [])
    self.assertEqual(target.inputs, [])

  def test_link_cleanup_shares_coordinator_history(self) -> None:
    """The coordinator retains each endpoint failure, not another wrapper."""

    source, target = TestBlock(), TestBlock()
    link = Link(source, target)
    link.close()
    inlet, outlet = Mock(), Mock()
    link._in, link._out = inlet, outlet
    link._in_closed = link._out_closed = False
    errors = (OSError('input close'), KeyboardInterrupt('output close'))
    inlet.close.side_effect, outlet.close.side_effect = errors
    manager = FailureManager('main', 'framework cleanup')
    manager._logger = Mock()
    with self.assertRaises(BaseExceptionGroup) as caught:
      with manager.collect('Block cleanup'):
        with manager.step('close Link'):
          link.close(manager=manager)
    self.assertEqual(exception_leaves(caught.exception), errors)
    self.assertEqual(len(manager.snapshot()), 2)
    self.assertEqual(manager._logger.error.call_count, 2)
    inlet.close.side_effect = outlet.close.side_effect = None
    link.close(manager=manager)
    link.close(manager=manager)
    self.assertEqual(inlet.close.call_count, 2)
    self.assertEqual(outlet.close.call_count, 2)

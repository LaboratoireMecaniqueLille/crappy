# coding: utf-8

"""Headless fault-injection checks for software-owned shutdown resources."""

from threading import Thread
from multiprocessing import Event
from unittest.mock import Mock, patch

import crappy.blocks.grapher as grapher_module
from crappy.blocks.camera_processes.display import Displayer
from crappy.blocks.grapher import Grapher
from crappy._failure import FailureManager
from crappy.links import Link
from crappy.links import link_graph

from tests.unit.blocks.test_gui_backends import GUI_BLOCKS, GUIBlockTestBase
from tests.support.assertions import exception_leaves, with_failure_manager
from tests.support.blocks import TestBlock


class TestSoftwareCleanup(GUIBlockTestBase):
  """Checks independent cleanup steps, error identity, and safe retries."""

  def test_gui_failures_do_not_skip_other_windows_or_events(self) -> None:
    """All GUI resources are attempted before several errors are raised."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = with_failure_manager(block_type(*args))
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
        self.assertGreaterEqual(app.processEvents.call_count, 2)
        app.quit.assert_not_called()

  def test_gui_interrupt_retains_its_cause_and_other_failures(self) -> None:
    """Interrupts remain group members without changing their causes."""

    block_type, _, args = GUI_BLOCKS[0]
    block = with_failure_manager(block_type(*args))
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
        block = with_failure_manager(block_type(*args))
        window, app = Mock(), Mock()
        window.close.return_value = False
        block._qt_window, block._qt_app = window, app
        with self.assertRaises(ExceptionGroup) as caught:
          block.finish()
        self.assertIsInstance(exception_leaves(caught.exception)[0],
                              RuntimeError)
        app.processEvents.assert_called()
        window.close.return_value = True
        block.finish()
        block.finish()
        self.assertEqual(window.close.call_count, 2)
        self.assertGreaterEqual(app.processEvents.call_count, 2)

  def test_grapher_closes_matplotlib_after_qt_failure(self) -> None:
    """A Qt close cannot skip event processing or Matplotlib cleanup."""

    with patch.object(grapher_module, 'warn'):
      block = with_failure_manager(Grapher(('x', 'y'), plotter='mpl', backend='TkAgg'))
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

    process = with_failure_manager(Displayer('cleanup', 10, backend='cv2'))
    process._stop_event = Event()
    process._to_draw_conn = Mock()

    def close_during_receive() -> None:
      process._stop_thread = True
      raise EOFError

    process._to_draw_conn.poll.return_value = True
    process._to_draw_conn.recv.side_effect = close_during_receive
    process._thread_target()
    self.assertTrue(process._stop_thread)

  def test_displayer_joins_thread_after_window_failure(self) -> None:
    """A failing display backend cannot skip overlay-thread cleanup."""

    process = with_failure_manager(Displayer('cleanup', 10, backend='cv2'))
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

    process = with_failure_manager(Displayer('cleanup', 10, backend='cv2'))
    error = RuntimeError('thread start')
    thread = Mock(spec=Thread)
    thread.start.side_effect = error
    with patch('crappy.blocks.camera_processes.display.Thread',
               return_value=thread):
      with self.assertRaises(RuntimeError):
        process.init()
    process.finish()
    thread.join.assert_not_called()

  def test_link_closes_both_endpoints_and_retries_only_failures(self) -> None:
    """Links release both endpoints without repeating a successful close."""

    source, target = TestBlock(), TestBlock()
    link = Link(source, target)
    with self.assertRaises(TypeError):
      link.close()
    manager = FailureManager('link owner', 'finish')
    manager._logger = Mock()
    link._in.close()
    link._out.close()
    inlet, outlet = Mock(), Mock()
    link._in, link._out = inlet, outlet
    error = OSError('input close')
    inlet.close.side_effect = error
    with self.assertRaises(ExceptionGroup) as caught:
      link.close(manager)
    self.assertEqual(exception_leaves(caught.exception), (error,))
    outlet.close.assert_called_once_with()
    inlet.close.side_effect = None
    link.close(manager)
    link.close(manager)
    self.assertEqual(inlet.close.call_count, 2)
    outlet.close.assert_called_once_with()

  def test_owned_cleanup_reports_without_outer_duplicate(self) -> None:
    """The Block lifecycle and its cleanup share one reporting history."""

    for block_type, _, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = with_failure_manager(block_type(*args))
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
        block = with_failure_manager(block_type(*args))
        window, app = Mock(), Mock()
        block._qt_window, block._qt_app = window, app
        error = OSError('progress log')
        with patch.object(block, 'log', side_effect=error):
          with self.assertRaises(ExceptionGroup) as caught:
            block.finish()
        self.assertEqual(exception_leaves(caught.exception), (error,))
        window.close.assert_called_once_with()
        app.processEvents.assert_called()

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
    manager = FailureManager('main', 'framework cleanup')
    manager._logger = Mock()
    link = Link(source, target)
    link.close(manager)
    inlet, outlet = Mock(), Mock()
    link._in, link._out = inlet, outlet
    link._in_closed = link._out_closed = False
    errors = (OSError('input close'), KeyboardInterrupt('output close'))
    inlet.close.side_effect, outlet.close.side_effect = errors
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

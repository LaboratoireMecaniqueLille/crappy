# coding: utf-8

"""Queue transfer and source-only reporting, without hardware dependencies."""

from dataclasses import replace
from io import StringIO
from multiprocessing import get_all_start_methods, get_context
import pickle
from queue import Queue
import subprocess
import sys
from threading import Event, Thread
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import Mock, patch

from crappy import Block
from crappy._exceptions import FailureTransportError
from crappy._failure import FailureManager, FailureReport


def _capture_worker(queue, ready, release, text):
  """Produces a report before its simulated finish is allowed to complete."""

  manager = FailureManager('Block', 'main', owner_path='Block.worker')
  manager._logger = Mock()
  manager.set_queue(queue)
  manager.capture_exception(ValueError(text))
  ready.set()
  if not release.wait(10):
    raise RuntimeError('test did not release worker finish')
  manager.set_queue(None)
  queue.close()
  queue.join_thread()


def _exercise_context(method):
  """Checks real delivery and flushing in an isolated start-method scenario."""

  context = get_context(method)
  manager = FailureManager('main', 'main')
  manager._logger = Mock()
  queues = tuple(context.Queue() for _ in range(2))
  ready = tuple(context.Event() for _ in queues)
  release = context.Event()
  stop = Event()
  Block.run_failure_manager = manager
  Block.failure_queues = queues
  Block.failure_thread_stop = stop
  Block.failure_thread = Thread(target=Block._failure_target, daemon=True)
  thread = Block.failure_thread
  processes = tuple(context.Process(
      target=_capture_worker, args=(queue, event, release, 'x' * 1048576))
      for queue, event in zip(queues, ready))
  try:
    for process in processes:
      process.start()
    thread.start()
    assert all(event.wait(10) for event in ready)
    deadline = monotonic() + 10
    while len(manager.snapshot()) < 2 and monotonic() < deadline:
      sleep(0.01)
    assert len(manager.snapshot()) == 2, manager.snapshot()
    assert all(process.is_alive() for process in processes)
    assert all(error is None for _, error in manager.snapshot())
    assert len({report.failure_id for report, _ in manager.snapshot()}) == 2
    manager._logger.error.assert_not_called()
    release.set()
    for process in processes:
      process.join(10)
      assert process.exitcode == 0, process.exitcode
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    try:
      manager.raise_if_failed('remote failures')
    except ExceptionGroup as errors:
      assert all(type(error) is ValueError for error in errors.exceptions)
      assert len(errors.exceptions) == 2
    else:
      raise AssertionError('remote failures were not raised')
  finally:
    release.set()
    for process in processes:
      if process.pid is not None:
        if process.is_alive():
          process.kill()
        process.join(2)
    stop.set()
    if thread.ident is not None:
      thread.join(2)
    for queue in queues:
      queue.close()
    Block.reset()


def _exercise_killed_writer(method):
  """Checks bounded joining when a killed producer leaves a partial read."""

  context = get_context(method)
  queue = context.Queue()
  ready, release = context.Event(), context.Event()
  process = context.Process(target=_capture_worker,
                            args=(queue, ready, release, 'x' * 4194304))
  manager = FailureManager('main', 'main')
  manager._logger = Mock()
  stop = Event()
  Block.run_failure_manager = manager
  Block.failure_queues = (queue,)
  Block.failure_thread_stop = stop
  Block.failure_thread = Thread(target=Block._failure_target, daemon=True)
  thread = Block.failure_thread
  try:
    process.start()
    assert ready.wait(10)
    # The writer has started filling the pipe; there is no reader yet.
    assert queue._reader.poll(5)
    process.kill()
    process.join(3)
    assert process.exitcode != 0
    thread.start()
    sleep(0.05)
    stop.set()
    started = monotonic()
    thread.join(0.1)
    assert monotonic() - started < 1
    # Public Queue closure does not promise to release a corrupted read.
    # This daemon can be abandoned until the test subprocess exits.
    queue.close()
    Block.reset()
  finally:
    # Killing a waiter can corrupt Event's condition counters. Do not notify
    # that discarded test Event afterwards; the producer is being killed.
    if process.pid is not None:
      if process.is_alive():
        process.kill()
      process.join(2)
    stop.set()
    queue.close()
    Block.reset()


class TestFailureTransport(TestCase):
  def setUp(self):
    """Connects source, forwarding owner, and root with inspectable Queues."""

    self.source = FailureManager('Block', 'main')
    self.parent = FailureManager('parent', 'main')
    self.root = FailureManager('main', 'main')
    for manager in (self.source, self.parent, self.root):
      manager._logger = Mock()
    self.incoming = Queue()
    self.outgoing = Queue()
    self.source.set_queue(self.incoming)
    self.parent.set_queue(self.outgoing)

  def test_capture_receive_and_silent_forwarding(self):
    report = self.source.capture_exception(ValueError('origin'))
    assert report is not None
    self.assertTrue(self.parent.receive((self.incoming,)))
    self.assertEqual(self.parent.snapshot(), ((report, None),))
    self.parent.forward(report)
    self.parent.forward(report, ValueError('restored'))
    self.assertEqual(self.outgoing.qsize(), 1)
    self.assertTrue(self.root.receive((self.outgoing,)))
    self.assertEqual(self.root.snapshot(), ((report, None),))
    self.assertFalse(self.root.receive((self.outgoing,)))
    self.source._logger.error.assert_called_once_with(str(report))
    self.parent._logger.error.assert_not_called()
    self.root._logger.error.assert_not_called()

  def test_payload_contains_only_serialized_wire_values(self):
    report = self.source.capture_exception(OSError(2, 'missing', 'input'))
    payload = self.incoming.get_nowait()
    self.assertIs(type(payload), bytes)
    wire = pickle.loads(payload)
    self.assertIs(type(wire), dict)
    self.assertEqual(FailureReport.from_wire(wire), report)

  def test_logger_failure_does_not_skip_original_or_error_transfer(self):
    error = OSError('logger failed')
    self.source._logger.error.side_effect = error
    with patch('crappy._failure.sys.stderr', StringIO()):
      with self.assertRaises(OSError) as caught:
        self.source.capture_exception(ValueError('origin'))
    self.assertIs(caught.exception, error)
    while self.parent.receive((self.incoming,)):
      pass
    self.assertEqual([report.exception_type
                      for report, _ in self.parent.snapshot()],
                     ['ValueError', 'OSError'])
    self.parent._logger.error.assert_not_called()

  def test_failed_stderr_does_not_skip_transfer(self):
    self.source._logger.error.side_effect = OSError('logger')
    emergency = Mock()
    emergency.write.side_effect = RuntimeError('stderr')
    with patch('crappy._failure.sys.stderr', emergency):
      with self.assertRaises(ExceptionGroup):
        self.source.capture_exception(ValueError('origin'))
    while self.parent.receive((self.incoming,)):
      pass
    self.assertEqual(len(self.parent.snapshot()), 3)

  def test_send_failure_retains_origin_and_disables_route(self):
    queue = Mock()
    error = OSError('send failed')
    queue.put.side_effect = error
    self.source.set_queue(queue)
    with self.assertRaises(OSError) as caught:
      self.source.capture_exception(ValueError('origin'))
    self.assertIs(caught.exception, error)
    self.assertEqual(len(self.source.snapshot()), 2)
    self.assertIs(self.source.snapshot()[1][1], error)
    self.source.capture_exception(RuntimeError('later'))
    queue.put.assert_called_once()
    self.assertEqual(self.source._logger.error.call_count, 3)

  def test_both_routes_fail_without_shadowing_origin(self):
    logger_error, send_error = OSError('logger'), RuntimeError('send')
    self.source._logger.error.side_effect = logger_error
    queue = Mock()
    queue.put.side_effect = send_error
    self.source.set_queue(queue)
    with patch('crappy._failure.sys.stderr', StringIO()):
      with self.assertRaises(ExceptionGroup) as caught:
        self.source.capture_exception(ValueError('origin'))
    self.assertEqual(caught.exception.exceptions, (logger_error, send_error))
    self.assertEqual([type(error) for _, error in self.source.snapshot()],
                     [ValueError, OSError, RuntimeError])
    self.source._logger.error.assert_called_once()
    queue.put.assert_called_once()

  def test_malformed_record_does_not_stop_other_children(self):
    self.incoming.put(b'not a pickle')
    other = Queue()
    self.source.set_queue(other)
    report = self.source.capture_exception(ValueError('origin'))
    assert report is not None
    self.assertTrue(self.parent.receive((self.incoming, other)))
    self.assertIsInstance(self.parent.snapshot()[0][1], FailureTransportError)
    self.assertEqual(self.parent.snapshot()[1], (report, None))
    self.incoming.put(pickle.dumps(report.to_wire()))
    self.assertTrue(self.parent.receive((self.incoming,)))
    self.assertEqual(len(self.parent.snapshot()), 2)

  def test_invalid_wire_and_nonbytes_payloads_are_reported(self):
    for value in ('unexpected', pickle.dumps({'nodes': []})):
      self.incoming.put(value)
      self.assertTrue(self.parent.receive((self.incoming,)))
    self.assertTrue(all(isinstance(error, FailureTransportError)
                        for _, error in self.parent.snapshot()))

  def test_receive_interruption_is_not_relabelled_as_an_ordinary_error(self):
    self.source.capture_exception(ValueError('origin'))
    interruption = KeyboardInterrupt('receive interrupted')
    with patch('crappy._failure.pickle.loads', side_effect=interruption):
      self.assertTrue(self.parent.receive((self.incoming,)))
    self.assertIs(self.parent.snapshot()[0][1], interruption)

  def test_failed_receiver_is_skipped_without_stopping_other_children(self):
    failed = Mock()
    failed.get_nowait.side_effect = OSError('receive failed')
    report = self.source.capture_exception(ValueError('origin'))
    self.parent.receive((failed, self.incoming))
    self.parent.receive((failed, self.incoming))
    failed.get_nowait.assert_called_once()
    self.assertEqual(self.parent.snapshot()[1], (report, None))

  def test_conflicting_id_is_a_transport_error(self):
    report = self.source.capture_exception(ValueError('origin'))
    assert report is not None
    self.parent.receive((self.incoming,))
    bad = replace(report, phase='different')
    self.incoming.put(pickle.dumps(bad.to_wire()))
    self.parent.receive((self.incoming,))
    self.assertEqual(self.parent.snapshot()[0], (report, None))
    self.assertIsInstance(self.parent.snapshot()[1][1], FailureTransportError)

  def test_simultaneous_captures_and_duplicate_forwards(self):
    report = self.source.capture_exception(ValueError('origin'))
    threads = [Thread(target=self.parent.forward, args=(report,))
               for _ in range(10)]
    threads += [Thread(target=self.source.capture_exception,
                       args=(ValueError(str(index)),)) for index in range(10)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join(2)
      self.assertFalse(thread.is_alive())
    self.assertEqual(self.outgoing.qsize(), 1)
    self.assertEqual(len(self.source.snapshot()), 11)
    self.assertEqual(self.incoming.qsize(), 11)

  def test_real_closed_queue_error_is_retained(self):
    """An error raised by the public put() call is retained with the origin."""

    queue = get_context().Queue()
    self.source.set_queue(queue)
    try:
      queue.close()
      with self.assertRaises(ValueError):
        self.source.capture_exception(ValueError('origin'))
      self.assertEqual(len(self.source.snapshot()), 2)
      self.assertEqual(self.source.snapshot()[1][0].phase, 'transport')
      self.assertIsNone(self.source._queue)
    finally:
      queue.close()
      queue.join_thread()

  def test_queue_setup_only_requires_the_public_api(self):
    """Connecting a sink does not install a private feeder callback."""

    queue = Mock(spec_set=('put', 'close', 'join_thread'))
    self.source.set_queue(queue)
    self.source.capture_exception(ValueError('origin'))
    queue.put.assert_called_once()

  def test_disconnect_stops_sending_without_skipping_retention(self):
    """Shutdown can detach the upstream Queue before its public closure."""

    self.source.capture_exception(ValueError('origin'))
    self.source.set_queue(None)
    self.source.capture_exception(RuntimeError('shutdown failure'))
    self.assertEqual(self.incoming.qsize(), 1)
    self.assertEqual(len(self.source.snapshot()), 2)
    self.assertEqual(self.source._logger.error.call_count, 2)

  def test_real_delivery_in_available_start_methods(self):
    script = ('import sys; from tests.framework.test_transport import '
              '_exercise_context; _exercise_context(sys.argv[1])')
    for method in get_all_start_methods():
      with self.subTest(method=method):
        result = subprocess.run([sys.executable, '-c', script, method],
                                capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_killed_writer_shutdown_in_available_start_methods(self):
    script = ('import sys; from tests.framework.test_transport import '
              '_exercise_killed_writer; _exercise_killed_writer(sys.argv[1])')
    for method in get_all_start_methods():
      with self.subTest(method=method):
        result = subprocess.run([sys.executable, '-c', script, method],
                                capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

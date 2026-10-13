# coding: utf-8

from multiprocessing import Barrier, Event, Pipe
from unittest.mock import Mock, call, patch

from crappy._exceptions import PrepareError
from crappy._failure import FailureManager
from crappy.blocks.vision.block import ConfigRequest
from crappy.tool.camera_config.tkinter import TkinterCameraConfig

from .vision_test_base import StubVisionBlock, VisionTestBase
from tests.block.block_test_base import exception_leaves


class TestVisionBlockConfiguration(VisionTestBase):
  """Tests configuration-request registration and Pipe exchange."""

  @staticmethod
  def make_request(connection=None,
                   source: str = 'source',
                   required: bool = True) -> ConfigRequest:
    """Creates a deterministic configuration request."""

    return ConfigRequest(requester='consumer',
                         args=(1, 2),
                         kwargs={'option': 3},
                         configurator=TkinterCameraConfig,
                         img_source=source,
                         connection=connection,
                         required=required)

  def test_request_registration_validates_type_and_connection(self) -> None:
    """Checks validation for incoming and outgoing request lists."""

    block = StubVisionBlock()

    for method in (block.add_config_request_in,
                   block.add_config_request_out):
      with self.subTest(method=method.__name__):
        with self.assertRaises(TypeError):
          method(object())
        with self.assertRaises(RuntimeError):
          method(self.make_request())

    incoming = self.make_request(connection=Mock())
    outgoing = self.make_request(connection=Mock())
    block.add_config_request_in(incoming)
    block.add_config_request_out(outgoing)

    self.assertEqual(block.config_requests_in, [incoming])
    self.assertEqual(block._config_requests_out, [outgoing])

  def test_prepare_rejects_unanswered_incoming_request(self) -> None:
    """Checks that a source cannot prepare with a pending response."""

    block = StubVisionBlock()
    request = self.make_request(connection=Mock())
    block.add_config_request_in(request)

    with self.assertRaises(RuntimeError):
      block.prepare()

    request.completed = True
    block.prepare()

  def test_send_config_sends_marks_complete_and_closes(self) -> None:
    """Checks successful required and optional configuration responses."""

    block = StubVisionBlock()

    for required, value in ((True, ('configured',)), (False, None)):
      with self.subTest(required=required):
        recv_conn, send_conn = Pipe(duplex=False)
        self.track_connection(recv_conn)
        self.track_connection(send_conn)
        request = self.make_request(connection=send_conn,
                                    required=required)

        block.send_config(request, value)

        self.assertTrue(request.completed)
        self.assertTrue(send_conn.closed)
        self.assertTrue(recv_conn.poll())
        self.assertEqual(recv_conn.recv(), value)

  def test_send_config_rejects_none_for_required_request(self) -> None:
    """Checks that declining a required request fails and closes its Pipe."""

    block = StubVisionBlock()
    recv_conn, send_conn = Pipe(duplex=False)
    self.track_connection(recv_conn)
    self.track_connection(send_conn)
    request = self.make_request(connection=send_conn, required=True)

    with self.assertRaises(ExceptionGroup) as caught:
      block.send_config(request, None)
    self.assertIsInstance(caught.exception.exceptions[0], RuntimeError)

    self.assertFalse(request.completed)
    self.assertTrue(send_conn.closed)

  def test_send_config_closes_connection_when_send_fails(self) -> None:
    """Checks cleanup when serializing or sending a response fails."""

    block = StubVisionBlock()
    connection = Mock()
    connection.send.side_effect = OSError('broken pipe')
    request = self.make_request(connection=connection)

    with self.assertRaises(ExceptionGroup) as caught:
      block.send_config(request, ('configured',))
    self.assertIsInstance(caught.exception.exceptions[0], OSError)

    self.assertFalse(request.completed)
    connection.close.assert_called_once_with()

  def test_recv_configs_collects_all_responses_and_closes(self) -> None:
    """Checks receiving configurations from multiple image sources."""

    block = StubVisionBlock()
    self.set_prepare_sync(block)

    for source, value in (('first', (1,)), ('second', None)):
      recv_conn, send_conn = Pipe(duplex=False)
      self.track_connection(recv_conn)
      self.track_connection(send_conn)
      send_conn.send(value)
      block.add_config_request_out(
          self.make_request(connection=recv_conn,
                            source=source,
                            required=value is not None))

    self.assertEqual(block.recv_configs(), {'first': (1,), 'second': None})
    self.assertTrue(all(request.connection.closed
                        for request in block._config_requests_out))

  def test_recv_configs_validates_startup_synchronization(self) -> None:
    """Checks the required preparation Barrier and stop Event."""

    block = StubVisionBlock()

    with self.assertRaises(ValueError):
      block.recv_configs()

    block._ready_barrier = Barrier(1)
    with self.assertRaises(ValueError):
      block.recv_configs()

  def test_recv_configs_stops_when_preparation_is_aborted(self) -> None:
    """Checks that waiting requests notice another Block's failure."""

    block = StubVisionBlock()
    block._ready_barrier = Barrier(1)
    block._stop_event = Event()
    block._stop_event.set()
    connection = Mock()
    connection.poll.return_value = False
    block.add_config_request_out(self.make_request(connection=connection))

    with self.assertRaises(ExceptionGroup) as caught:
      block.recv_configs()
    self.assertIsInstance(caught.exception.exceptions[0], PrepareError)

    connection.poll.assert_called_once_with(timeout=0.5)
    connection.close.assert_called_once_with()

  def test_recv_configs_converts_broken_pipe_to_prepare_error(self) -> None:
    """Checks the error reported when a source exits without responding."""

    block = StubVisionBlock()
    self.set_prepare_sync(block)
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.side_effect = EOFError
    block.add_config_request_out(self.make_request(connection=connection))

    with self.assertRaises(ExceptionGroup) as caught:
      block.recv_configs()
    self.assertIsInstance(caught.exception.exceptions[0], PrepareError)

    connection.close.assert_called_once_with()

  def test_exchange_failure_keeps_pipe_close_interrupt(self) -> None:
    """Request cleanup retains its failure alongside the exchange failure."""

    for send in (True, False):
      with self.subTest(send=send):
        block = StubVisionBlock()
        self.set_prepare_sync(block)
        manager = FailureManager(block.name, 'prepare')
        block._failure_manager = manager
        manager._logger = Mock()
        connection = Mock()
        origin, cleanup = ValueError('exchange'), KeyboardInterrupt('close')
        connection.send.side_effect = connection.recv.side_effect = origin
        connection.poll.return_value = True
        connection.close.side_effect = cleanup
        request = self.make_request(connection=connection)
        with self.assertRaises(BaseExceptionGroup) as caught:
          if send:
            block.send_config(request, ('configured',))
          else:
            block.add_config_request_out(request)
            block.recv_configs()
        self.assertEqual(exception_leaves(caught.exception), (origin, cleanup))
        self.assertFalse(request.completed)
        self.assertFalse(request.connection_closed)
        connection.close.assert_called_once_with()
        self.assertEqual(manager._logger.error.call_count, 2)
        connection.close.side_effect = None
        block.finish()

  def test_send_diagnostic_failure_keeps_completed_flag(self) -> None:
    """A sent response stays completed even if its success log fails."""

    block = StubVisionBlock()
    connection = Mock()
    request = self.make_request(connection=connection)
    error = OSError('diagnostic')
    with patch.object(block, 'log', side_effect=error):
      with self.assertRaises(ExceptionGroup) as caught:
        block.send_config(request, ('configured',))
    self.assertEqual(exception_leaves(caught.exception), (error,))
    self.assertTrue(request.completed)
    self.assertTrue(request.connection_closed)
    connection.close.assert_called_once_with()

  def test_finish_closes_unanswered_config_pipes(self) -> None:
    """Preparation can fail before any configuration response is exchanged."""

    block = StubVisionBlock()
    incoming, outgoing = Pipe(duplex=False)
    self.track_connection(incoming)
    self.track_connection(outgoing)
    block.add_config_request_in(self.make_request(connection=outgoing))
    block.add_config_request_out(self.make_request(connection=incoming))

    block.finish()
    block.finish()

    self.assertTrue(incoming.closed)
    self.assertTrue(outgoing.closed)

  def test_finish_continues_after_pipe_close_failure(self) -> None:
    """All remaining endpoints are closed before reporting a failed close."""

    block = StubVisionBlock()
    operations = Mock()
    incoming, outgoing = Mock(), Mock()
    operations.attach_mock(incoming, 'incoming')
    operations.attach_mock(outgoing, 'outgoing')
    error = RuntimeError('close failed')
    outgoing.close.side_effect = error
    block.add_config_request_in(self.make_request(connection=incoming))
    block.add_config_request_out(self.make_request(connection=outgoing))

    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertIn('configuration Pipe', error.__notes__[0])
    self.assertEqual(operations.mock_calls,
                     [call.incoming.close(), call.outgoing.close()])
    outgoing.close.side_effect = None
    block.finish()
    block.finish()
    incoming.close.assert_called_once_with()
    self.assertEqual(outgoing.close.call_count, 2)

  def test_finish_does_not_reclose_exchanged_config_pipes(self) -> None:
    """Pipes already closed by send/receive are not closed twice."""

    block = StubVisionBlock()
    self.set_prepare_sync(block)
    incoming, outgoing = Mock(), Mock()
    outgoing.poll.return_value = True
    outgoing.recv.return_value = ('configured',)
    sent = self.make_request(connection=incoming)
    block.add_config_request_in(sent)
    block.add_config_request_out(self.make_request(connection=outgoing))
    block.send_config(sent, ('configured',))
    block.recv_configs()
    block.finish()
    incoming.close.assert_called_once_with()
    outgoing.close.assert_called_once_with()


if __name__ == '__main__':
  import unittest
  unittest.main()

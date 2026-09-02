# coding: utf-8

from crappy._global import LinkDataError
from crappy.tool.camera_config import Box

from .camera_process_test_base import (CameraProcessTestBase, TestLink,
                                       TestCameraProcess)


class TestLinks(CameraProcessTestBase):
  """Tests CameraProcess helpers for outgoing data and overlays."""

  def test_send(self) -> None:
    """Tests the different accepted inputs of CameraProcess.send."""

    self._process = TestCameraProcess()
    links = [TestLink("link_1"), TestLink("link_2")]
    self.make_shared(outputs=links, labels=None)
    self.set_test_logger()

    self._process.send(None)
    for link in links:
      with self.subTest(link=link.name):
        self.assertFalse(link.sent.is_set())
        self.assertEqual(link.sent_values, list())

    with self.assertRaises(LinkDataError):
      self._process.send(0)

    self._process.send({'a': 0, 'b': 1})
    for link in links:
      with self.subTest(link=link.name):
        self.assertTrue(link.sent.is_set())
        self.assertEqual(link.sent_values[-1], {'a': 0, 'b': 1})
        link.sent.clear()

    self._process._labels = ['a', 'b']
    self._process.send((2, 3))
    for link in links:
      with self.subTest(link=link.name):
        self.assertTrue(link.sent.is_set())
        self.assertEqual(link.sent_values[-1], {'a': 2, 'b': 3})
        link.sent.clear()

    # Extra labels follow zip() semantics and are ignored when no value exists.
    self._process._labels = ['a', 'b', 'c']
    self._process.send((4, 5))
    for link in links:
      with self.subTest(link=link.name):
        self.assertTrue(link.sent.is_set())
        self.assertEqual(link.sent_values[-1], {'a': 4, 'b': 5})

  def test_send_to_draw(self) -> None:
    """Tests sending overlays to the Displayer process."""

    self._process = TestCameraProcess()
    overlay = Box(x_start=1, x_end=5, y_start=2, y_end=6)

    # Missing Connection should be a no-op.
    self._process.send_to_draw([overlay])

    recv_conn, send_conn = self.make_pipe()
    self._process._to_draw_conn = send_conn

    self._process.send_to_draw([overlay])

    self.assertTrue(recv_conn.poll(1.0))
    received = recv_conn.recv()

    self.assertEqual(len(received), 1)
    self.assertIsInstance(received[0], Box)
    self.assertEqual(received[0].x_start, 1)
    self.assertEqual(received[0].x_end, 5)
    self.assertEqual(received[0].y_start, 2)
    self.assertEqual(received[0].y_end, 6)

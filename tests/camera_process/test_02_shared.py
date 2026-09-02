# coding: utf-8

import logging
import numpy as np

from .camera_process_test_base import (CameraProcessTestBase, TestLink,
                                       TestCameraProcess)


class TestSharedObjects(CameraProcessTestBase):
  """Tests sharing Camera-owned multiprocessing objects."""

  def test_set_shared(self) -> None:
    """Tests shared references and local image allocation."""

    self._process = TestCameraProcess()
    recv_conn, send_conn = self.make_pipe()
    outputs = [TestLink()]
    labels = ['x', 'y']

    shared = self.make_shared(shape=(2, 3),
                              dtype=np.uint16,
                              to_draw_conn=send_conn,
                              outputs=outputs,
                              labels=labels,
                              log_level=logging.ERROR,
                              display_freq=True)

    self.assertIs(self._process._img_array, shared.array)
    self.assertIs(self._process._data_dict, shared.data_dict)
    self.assertIs(self._process._lock, shared.lock)
    self.assertIs(self._process._cam_barrier, shared.barrier)
    self.assertIs(self._process._stop_event, shared.stop_event)
    self.assertEqual(self._process._shape, (2, 3))
    self.assertEqual(self._process._dtype, np.dtype(np.uint16))
    self.assertIs(self._process._to_draw_conn, send_conn)
    self.assertIs(self._process._outputs, outputs)
    self.assertIs(self._process._labels, labels)
    self.assertIs(self._process._log_queue, shared.log_queue)
    self.assertEqual(self._process._log_level, logging.ERROR)
    self.assertTrue(self._process._display_freq)

    self.assertIsNotNone(self._process.img)
    self.assertEqual(self._process.img.shape, (2, 3))
    self.assertEqual(self._process.img.dtype, np.dtype(np.uint16))

    self.assertFalse(recv_conn.poll())

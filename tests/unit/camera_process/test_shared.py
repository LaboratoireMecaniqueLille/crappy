# coding: utf-8

import logging
from inspect import signature, Parameter
from unittest.mock import MagicMock
import numpy as np

from tests.support.camera_process_test_case import CameraProcessTestBase
from tests.support.camera_processes import TestLink, TestCameraProcess


class TestSharedObjects(CameraProcessTestBase):
  """Tests sharing Camera-owned multiprocessing objects."""

  def test_reporting_dependencies_are_required(self) -> None:
    parameters = signature(TestCameraProcess.set_shared).parameters
    for name in ('failure_queue', 'block_name'):
      with self.subTest(argument=name):
        self.assertIs(parameters[name].default, Parameter.empty)

  def test_condition_is_accepted_without_private_lock_inspection(self) -> None:
    """A caller-supplied Condition needs only its public interface."""

    self._process = TestCameraProcess()
    shared = self.make_shared()
    arguments = shared._asdict()
    arguments['event'] = arguments.pop('stop_event')
    arguments['block_name'] = 'test Camera'
    condition = MagicMock(spec_set=('wait_for', 'notify_all',
                                    '__enter__', '__exit__'))
    arguments['condition'] = condition
    self._process.set_shared(**arguments)
    self.assertIs(self._process._condition, condition)

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
    self.assertIs(self._process._condition, shared.condition)
    self.assertIs(self._process._cam_barrier, shared.barrier)
    self.assertIs(self._process._stop_event, shared.stop_event)
    self.assertEqual(self._process._shape, (2, 3))
    self.assertEqual(self._process._dtype, np.dtype(np.uint16))
    self.assertIs(self._process._to_draw_conn, send_conn)
    self.assertIs(self._process._outputs, outputs)
    self.assertIs(self._process._labels, labels)
    self.assertIs(self._process._log_queue, shared.log_queue)
    self.assertIs(self._process._failure_queue, shared.failure_queue)
    self.assertEqual(self._process._block_name, 'test Camera')
    self.assertEqual(self._process._log_level, logging.ERROR)
    self.assertTrue(self._process._display_freq)

    self.assertIsNotNone(self._process.img)
    self.assertEqual(self._process.img.shape, (2, 3))
    self.assertEqual(self._process.img.dtype, np.dtype(np.uint16))

    self.assertFalse(recv_conn.poll())

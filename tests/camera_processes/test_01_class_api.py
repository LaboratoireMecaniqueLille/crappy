# coding: utf-8

from multiprocessing import current_process
from platform import system
from crappy.blocks.camera_processes.camera_process import CameraProcess

from .camera_process_test_base import CameraProcessTestBase, TestCameraProcess


class TestClassAPI(CameraProcessTestBase):
  """Tests covering the basic CameraProcess object API."""

  def test_abstract_class(self) -> None:
    """Tests that CameraProcess cannot be instantiated directly."""

    with self.assertRaises(TypeError):
      CameraProcess()

  def test_init_defaults(self) -> None:
    """Tests the default state of a freshly instantiated CameraProcess."""

    self._process = TestCameraProcess()

    self.assertEqual(self._process.name,
                     f"{current_process().name}.TestCameraProcess")
    self.assertEqual(self._process._system, system())

    self.assertIsNone(self._process._log_queue)
    self.assertIsNone(self._process._logger)
    self.assertIsNone(self._process._log_level)

    self.assertIsNone(self._process._img_array)
    self.assertIsNone(self._process._data_dict)
    self.assertIsNone(self._process._lock)
    self.assertIsNone(self._process._cam_barrier)
    self.assertIsNone(self._process._stop_event)
    self.assertIsNone(self._process._shape)
    self.assertIsNone(self._process._to_draw_conn)
    self.assertEqual(self._process._outputs, list())
    self.assertEqual(self._process._labels, list())
    self.assertIsNone(self._process.img)
    self.assertIsNone(self._process._dtype)
    self.assertEqual(self._process.metadata, {'ImageUniqueID': None})
    self.assertFalse(self._process._img0_set)

    self.assertEqual(self._process.fps_count, 0)
    self.assertIsNone(self._process._display_freq)

# coding: utf-8

import numpy as np
from crappy.blocks.camera_processes.camera_process import CameraProcess

from tests.support.camera_process_test_case import CameraProcessTestBase
from tests.support.camera_processes import TestCameraProcess


class TestClassAPI(CameraProcessTestBase):
  """Tests covering the basic CameraProcess object API."""

  def test_abstract_class(self) -> None:
    """Tests that CameraProcess cannot be instantiated directly."""

    with self.assertRaises(TypeError):
      CameraProcess()

  def test_default_process_receives_frame_and_finishes(self) -> None:
    """A default process can receive a frame and finish its lifecycle."""

    self._process = TestCameraProcess()
    shared = self.make_shared()
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    metadata = {'ImageUniqueID': 7, 't(s)': 1.5}
    self.write_image(shared, image, metadata)

    self._process.run()

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
    np.testing.assert_array_equal(self._process.img, image)
    self.assertEqual(self._process.metadata, metadata)
    self.assertFalse(self._process._failure_manager.snapshot())

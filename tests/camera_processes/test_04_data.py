# coding: utf-8

import numpy as np

from .camera_process_test_base import CameraProcessTestBase, TestCameraProcess


class TestData(CameraProcessTestBase):
  """Tests image and metadata transfer from shared objects."""

  def test_get_data(self) -> None:
    """Tests _get_data on missing, new, repeated and updated frames."""

    self._process = TestCameraProcess()
    shared = self.make_shared(shape=(2, 3), dtype=np.uint16)

    self.assertFalse(self._process._get_data())

    img = np.arange(6, dtype=np.uint16).reshape(2, 3)
    metadata = {'ImageUniqueID': 1, 't(s)': 1.0, 'meta': 'first'}
    self.write_image(shared, img, metadata)

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata, metadata)
    np.testing.assert_array_equal(self._process.img, img)

    # The same frame should not be handled twice.
    self.assertFalse(self._process._get_data())

    img_2 = img + 10
    metadata_2 = {'ImageUniqueID': 2, 't(s)': 2.0, 'meta': 'second'}
    self.write_image(shared, img_2, metadata_2)

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata, metadata_2)
    np.testing.assert_array_equal(self._process.img, img_2)

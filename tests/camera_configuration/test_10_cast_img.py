# coding: utf-8

import numpy as np

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestCastImage(ConfigurationWindowTestBase):
  """Class for testing the behavior of the configuration window when facing 
  various image types.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def test_cast_image(self) -> None:
    """Tests whether the interface behaves as expected when provided with
    images of various data types and bit depths."""

    # First, check that all variables are uninitialized
    self.assertIsNone(self._config._original_img)
    self.assertIsNone(self._config._img)
    self.assertIsNone(self._config._pil_img)

    # From this base we'll create a variety of images to test
    x, y = np.mgrid[0:240, 0:320]

    # 1 Channel, Shape 2, 8 bits image
    img_1_2_8 = ((x + y) / np.max(x + y) * 255).astype(np.uint8)

    # The "regular" grey level image should be untouched
    self._config._cast_img(img_1_2_8)
    self.assertTrue(np.all(self._config._original_img == img_1_2_8))
    self.assertTrue(np.all(self._config._img == img_1_2_8))

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 1 Channel, Shape 3, 8 bits image
    img_1_3_8 = ((x + y) / np.max(x + y) * 255).astype(np.uint8)
    img_1_3_8 = img_1_3_8[:, :, np.newaxis]

    # The 3 dimensions grey level image should be cast to 2 dimensions
    self._config._cast_img(img_1_3_8)
    self.assertTrue(np.all(self._config._original_img == img_1_3_8[:, :, 0]))
    self.assertTrue(np.all(self._config._img == img_1_3_8[:, :, 0]))

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 2 Channel, Shape 3, 8 bits image
    img_2_3_8 = ((x + y) / np.max(x + y) * 255).astype(np.uint8)
    img_2_3_8 = np.stack((img_2_3_8, img_2_3_8), axis=2)

    # The 3 dimensions grey level + alpha image should be cast to 2 dimensions
    # and the alpha channel removed
    self._config._cast_img(img_2_3_8)
    self.assertTrue(np.all(self._config._original_img == img_2_3_8[:, :, 0]))
    self.assertTrue(np.all(self._config._img == img_2_3_8[:, :, 0]))

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 3 Channel, Shape 3, 8 bits image
    img_3_3_8 = ((x + y) / np.max(x + y) * 255).astype(np.uint8)
    img_3_3_8 = np.stack((img_3_3_8, img_3_3_8 - 1, img_3_3_8 + 1),
                         axis=2)

    # The 3 dimensions RGB image should be reversed
    self._config._cast_img(img_3_3_8)
    self.assertTrue(np.all(self._config._original_img ==
                           img_3_3_8[:, :, ::-1]))
    self.assertTrue(np.all(self._config._img == img_3_3_8[:, :, ::-1]))

    # The PIL image should be in mode RGB
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'RGB')

    # 4 Channel, Shape 3, 8 bits image
    img_4_3_8 = ((x + y) / np.max(x + y) * 255).astype(np.uint8)
    img_4_3_8 = np.stack((img_4_3_8, img_4_3_8 - 1, img_4_3_8 + 1,
                          img_4_3_8), axis=2)

    # The 3 dimensions RGBA image should be reversed and have the alpha channel
    # removed
    self._config._cast_img(img_4_3_8)
    self.assertTrue(np.all(self._config._original_img ==
                           img_4_3_8[:, :, :3][:, :, ::-1]))
    self.assertTrue(np.all(self._config._img ==
                           img_4_3_8[:, :, :3][:, :, ::-1]))

    # The PIL image should be in mode RGB
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'RGB')

    # 1 Channel, Shape 2, 16 bits image
    img_1_2_16 = ((x + y) / np.max(x + y) * 255).astype(np.uint16)

    # The 16 bits grey level image should be cast to 8 bits
    self._config._cast_img(img_1_2_16)
    self.assertEqual(self._config._original_img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.max(), 255)
    self.assertEqual(self._config._img.min(), 0)

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 1 Channel, Shape 3, 16 bits image
    img_1_3_16 = ((x + y) / np.max(x + y) * 255).astype(np.uint16)
    img_1_3_16 = img_1_3_16[:, :, np.newaxis]

    # The 3 dimensions 16 bits grey level image should be cast to 2 dimensions
    # and 8 bits
    self._config._cast_img(img_1_3_16)
    self.assertEqual(self._config._original_img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.max(), 255)
    self.assertEqual(self._config._img.min(), 0)

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 2 Channel, Shape 3, 16 bits image
    img_2_3_16 = ((x + y) / np.max(x + y) * 255).astype(np.uint16)
    img_2_3_16 = np.stack((img_2_3_16, img_2_3_16), axis=2)

    # The 3 dimensions grey level + alpha 16 bits image should be cast to 2
    # dimensions and 8 bits, and the alpha channel removed
    self._config._cast_img(img_2_3_16)
    self.assertEqual(self._config._original_img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.max(), 255)
    self.assertEqual(self._config._img.min(), 0)

    # The PIL image should be in mode L
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'L')

    # 3 Channel, Shape 3, 16 bits image
    img_3_3_16 = ((x + y) / np.max(x + y) * 255).astype(np.uint16)
    img_3_3_16 = np.stack((img_3_3_16, img_3_3_16 + 1, img_3_3_16 - 1), axis=2)

    # The 3 dimensions RGB 16 bits image should be reversed and cast to 8 bits
    self._config._cast_img(img_3_3_16)
    self.assertEqual(self._config._original_img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.max(), 255)
    self.assertEqual(self._config._img.min(), 0)

    # The PIL image should be in mode RGB
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'RGB')

    # 4 Channel, Shape 3, 16 bits image
    img_4_3_16 = ((x + y) / np.max(x + y) * 255).astype(np.uint16)
    img_4_3_16 = np.stack((img_4_3_16, img_4_3_16 + 1, img_4_3_16 - 1,
                           img_4_3_16), axis=2)

    # The 3 dimensions RGBA 16 bits image should be reversed and cast to 8 bits
    # and have the alpha channel removed
    self._config._cast_img(img_4_3_16)
    self.assertEqual(self._config._original_img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.dtype, np.dtypes.UInt8DType())
    self.assertEqual(self._config._img.max(), 255)
    self.assertEqual(self._config._img.min(), 0)

    # The PIL image should be in mode RGB
    self._config._resize_img()
    self.assertEqual(self._config._pil_img.mode, 'RGB')

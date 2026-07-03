# coding: utf-8

from unittest import TestCase

import numpy as np

from crappy.tool.image_processing.fields import allowed_fields, get_field, get_res


class TestFields(TestCase):
  """Unit tests for the image-processing field helpers."""

  def test_all_allowed_fields_generate_two_float32_arrays(self) -> None:
    """Checks field array shape and dtype for all built-in fields."""

    for field in allowed_fields:
      with self.subTest(field=field):
        field_x, field_y = get_field(field, 3, 5)

        self.assertEqual(field_x.shape, (3, 5))
        self.assertEqual(field_y.shape, (3, 5))
        self.assertEqual(field_x.dtype, np.float32)
        self.assertEqual(field_y.dtype, np.float32)

  def test_basic_translation_fields(self) -> None:
    """Checks the simplest generated displacement fields."""

    field_x, field_y = get_field('x', 2, 3)

    np.testing.assert_array_equal(field_x, np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(field_y, np.zeros((2, 3), dtype=np.float32))

    field_x, field_y = get_field('y', 2, 3)

    np.testing.assert_array_equal(field_x, np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(field_y, np.ones((2, 3), dtype=np.float32))

  def test_strain_fields_are_scaled_by_image_size(self) -> None:
    """Checks the expected scaling of strain-like fields."""

    exx_x, exx_y = get_field('exx', 2, 3)
    eyy_x, eyy_y = get_field('eyy', 3, 2)

    np.testing.assert_allclose(exx_x[0], [-0.015, 0.0, 0.015])
    np.testing.assert_array_equal(exx_y, np.zeros((2, 3), dtype=np.float32))

    np.testing.assert_array_equal(eyy_x, np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_allclose(eyy_y[:, 0], [-0.015, 0.0, 0.015])

  def test_unknown_field_raises(self) -> None:
    """Checks validation for unknown field strings."""

    with self.assertRaises(NameError):
      get_field('missing', 2, 2)

  def test_get_res_is_zero_for_identical_images_and_zero_flow(self) -> None:
    """Checks residual calculation for the identity remapping."""

    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    flow = np.zeros((4, 4, 2), dtype=np.float32)

    np.testing.assert_allclose(get_res(img, img, flow), np.zeros((4, 4)))

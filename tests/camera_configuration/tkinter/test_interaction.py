# coding: utf-8

"""Tk mouse-wheel zoom and right-button panning."""

from itertools import product
from platform import system
from types import SimpleNamespace

from ._fixtures import TkinterConfigTestCase, FakeTestCameraSimple


class TestZoom(TkinterConfigTestCase):
  """Class for testing the zoom feature of the configuration window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple(min_val=0, max_val=255)

  def test_zoom(self) -> None:
    """Tests whether the image is correctly updates when zooming in and out."""

    self.run_config_cycle()

    # The zoom level should initially be set to 0
    self.assertEqual(self._config._zoom_step, 0)
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # The zoom commands differ on Linux, Windows, and macOS
    # Generating a single zoom-in event at 0, 0 with the mousewheel
    if system() == "Linux":
      self._config._img_canvas.event_generate('<4>', when="now", x=0, y=0)
    else:
      self._config._img_canvas.event_generate('<MouseWheel>', when="now",
                                              x=0, y=0, delta=1)

    # The zoom level should be unchanged as we're outside the image
    self.assertEqual(self._config._zoom_step, 0)
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # Generating a single zoom-in event at the image border with the mousewheel
    if system() == "Linux":
      self._config._img_canvas.event_generate(
          '<4>', when="now", x=self._config._img_canvas.winfo_width(),
          y=self._config._img_canvas.winfo_height())
    else:
      self._config._img_canvas.event_generate(
          '<MouseWheel>', when="now", x=self._config._img_canvas.winfo_width(),
          y=self._config._img_canvas.winfo_height(), delta=1)

    # The zoom level should be unchanged as we're outside the image
    self.assertEqual(self._config._zoom_step, 0)
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # Generating a single zoom-out event at the image center
    if system() == "Linux":
      self._config._img_canvas.event_generate(
          '<5>', when="now", x=self._config._img_canvas.winfo_width(),
          y=self._config._img_canvas.winfo_height())
    else:
      self._config._img_canvas.event_generate(
          '<MouseWheel>', when="now", x=self._config._img_canvas.winfo_width(),
          y=self._config._img_canvas.winfo_height(), delta=-1)

    # The zoom level should be unchanged as we're already zoomed-out
    self.assertEqual(self._config._zoom_step, 0)
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # Generating a single zoom-in event with the mousewheel in the image center
    if system() == "Linux":
      self._config._img_canvas.event_generate(
          '<4>', when="now", x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2)
    else:
      self._config._img_canvas.event_generate(
          '<MouseWheel>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2, delta=1)

    # Checking that the zoom parameters have been updated correctly
    self.assertEqual(self._config._zoom_step, 1)
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           (1 - self._config._zoom_ratio) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           (1 + self._config._zoom_ratio) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           (1 - self._config._zoom_ratio) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           (1 + self._config._zoom_ratio) / 2, delta=0.001)

    # Generating multiple zoom-in events in the image center
    for _ in range(self._config._max_zoom_step):
      if system() == "Linux":
        self._config._img_canvas.event_generate(
            '<4>', when="now", x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2)
      else:
        self._config._img_canvas.event_generate(
            '<MouseWheel>', when="now",
            x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2, delta=1)

    # The zoom step should be limited to the maximum allowed value
    self.assertEqual(self._config._zoom_step, self._config._max_zoom_step)
    # The zoom level can be computed via an explicit formula in this specific
    # case
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           (1 - self._config._zoom_ratio **
                            self._config._max_zoom_step) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           (1 + self._config._zoom_ratio **
                            self._config._max_zoom_step) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           (1 - self._config._zoom_ratio **
                            self._config._max_zoom_step) / 2, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           (1 + self._config._zoom_ratio **
                            self._config._max_zoom_step) / 2, delta=0.001)

    # Generating multiple zoom-out events in the image center
    for _ in range(self._config._max_zoom_step + 3):
      if system() == "Linux":
        self._config._img_canvas.event_generate(
            '<5>', when="now", x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2)
      else:
        self._config._img_canvas.event_generate(
            '<MouseWheel>', when="now",
            x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2, delta=-1)

    # The zoom level should be back to default
    self.assertEqual(self._config._zoom_step, 0)
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # Define values to use in test loop
    # Representative corners and interior points keep the Cartesian coverage
    # useful without performing hundreds of expensive Tk image redraws.
    to_test = (0, 85, 170, 255)
    img_ratio = 320 / 240

    # Loop over many possible zoom configurations to test if they all give the
    # expected displayed image
    for x_min, x_max, y_min, y_max in product(to_test, repeat=4):
      with self.subTest(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):

        # Only consider cases when the image is valid
        if x_min >= x_max or y_min >= y_max:
          continue

        # Set the zoom level to arbitrary values
        self._config._zoom_values.x_low = x_min / 255
        self._config._zoom_values.x_high = x_max / 255
        self._config._zoom_values.y_low = y_min / 255
        self._config._zoom_values.y_high = y_max / 255

        self.run_config_cycle()

        # Check that the displayed sub-image is the expected one
        min_, max_ = self._config._pil_img.getextrema()
        self.assertAlmostEqual(int((x_min / 255 * img_ratio + y_min / 255) /
                                   (img_ratio + 1) * 255), min_, delta=2)
        self.assertAlmostEqual(int((x_max / 255 * img_ratio + y_max / 255) /
                                   (img_ratio + 1) * 255), max_, delta=2)


class TestDrag(TkinterConfigTestCase):
  """Class for testing the drag feature of the configuration window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple(min_val=0, max_val=255)

  def test_drag(self) -> None:
    """Tests whether the image is correctly updates when dragging it around."""

    self.run_config_cycle()

    # The entire image should initially be displayed
    self.assertEqual(self._config._zoom_values.x_low, 0.0)
    self.assertEqual(self._config._zoom_values.x_high, 1.0)
    self.assertEqual(self._config._zoom_values.y_low, 0.0)
    self.assertEqual(self._config._zoom_values.y_high, 1.0)

    # The extrema of the image should be 0 and 255
    self.assertEqual(self._config._pil_img.getextrema(), (0, 255))

    # The movement variables should be None at that point
    self.assertIsNone(self._config._move_x)
    self.assertIsNone(self._config._move_y)

    # Click and drag the image
    self._config._img_canvas.event_generate('<ButtonPress-3>',
                                            when="now", x=100, y=100)
    self._config._img_canvas.event_generate('<B3-Motion>',
                                            when="now", x=50, y=50)

    # Should have no effect since the image is fully zoomed out
    self.assertEqual(self._config._pil_img.getextrema(), (0, 255))

    # The movement variables should have been set though
    self.assertIsNotNone(self._config._move_x)
    self.assertIsNotNone(self._config._move_y)

    # Generating multiple zoom-in events in the image center
    for _ in range(self._config._max_zoom_step):
      if system() == "Linux":
        self._config._img_canvas.event_generate(
            '<4>', when="now", x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2)
      else:
        self._config._img_canvas.event_generate(
            '<MouseWheel>', when="now",
            x=self._config._img_canvas.winfo_width() // 2,
            y=self._config._img_canvas.winfo_height() // 2, delta=1)

    # The zoom level can be computed analytically
    level_min = (1 - self._config._zoom_ratio **
                 self._config._max_zoom_step) / 2
    level_max = (1 + self._config._zoom_ratio **
                 self._config._max_zoom_step) / 2
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.001)

    # The displayed sub-image should be as expected
    img_ratio = 320 / 240
    min_, max_ = self._config._pil_img.getextrema()
    self.assertAlmostEqual(int((level_min * img_ratio + level_min) /
                               (img_ratio + 1) * 255), min_, delta=2)
    self.assertAlmostEqual(int((level_max * img_ratio + level_max) /
                               (img_ratio + 1) * 255), max_, delta=2)

    # Start outside the image, then move onto it. This must not reuse the
    # coordinates left by the previous drag.
    self._config._start_move(SimpleNamespace(x=-1, y=-1))
    self.assertIsNone(self._config._move_x)
    self.assertIsNone(self._config._move_y)
    center_x = self._config._img_canvas.winfo_width() // 2
    center_y = self._config._img_canvas.winfo_height() // 2
    self._config._img_canvas.event_generate('<B3-Motion>',
                                            when="now", x=center_x, y=center_y)

    # Should have no effect on the image
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.001)

    # Click and drag the image on its opposite corner
    self._config._img_canvas.event_generate(
        '<ButtonPress-3>', when="now",
        x=self._config._img_canvas.winfo_width(),
        y=self._config._img_canvas.winfo_height())
    self._config._img_canvas.event_generate('<B3-Motion>',
                                            when="now", x=50, y=50)

    # Should still have no effect on the image
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.001)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.001)

    diff_w = ((self._config._img_canvas.winfo_width() // 4) /
              self._config._img_canvas.winfo_width() *
              (level_max - level_min))
    diff_h = ((self._config._img_canvas.winfo_height() // 4) /
              self._config._img_canvas.winfo_height() *
              (level_max - level_min))

    # Dragging the image to the right
    self._config._img_canvas.event_generate(
        '<ButtonPress-3>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=self._config._img_canvas.winfo_height() // 2)
    self._config._img_canvas.event_generate(
        '<B3-Motion>', when="now",
        x=self._config._img_canvas.winfo_width() // 4,
        y=self._config._img_canvas.winfo_height() // 2)

    # The image should have been dragged
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min + diff_w, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max + diff_w, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.01)

    # Dragging the image to the bottom
    self._config._img_canvas.event_generate(
        '<ButtonPress-3>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=self._config._img_canvas.winfo_height() // 2)
    self._config._img_canvas.event_generate(
        '<B3-Motion>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=self._config._img_canvas.winfo_height() // 4)

    # The image should have been dragged
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min + diff_w, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max + diff_w, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min + diff_h, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max + diff_h, delta=0.01)

    # Dragging the image to the left
    self._config._img_canvas.event_generate(
        '<ButtonPress-3>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=self._config._img_canvas.winfo_height() // 2)
    self._config._img_canvas.event_generate(
        '<B3-Motion>', when="now",
        x=3 * self._config._img_canvas.winfo_width() // 4,
        y=self._config._img_canvas.winfo_height() // 2)

    # The image should have been dragged
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min + diff_h, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max + diff_h, delta=0.01)

    # Dragging the image to the top
    self._config._img_canvas.event_generate(
        '<ButtonPress-3>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=self._config._img_canvas.winfo_height() // 2)
    self._config._img_canvas.event_generate(
        '<B3-Motion>', when="now",
        x=self._config._img_canvas.winfo_width() // 2,
        y=3 * self._config._img_canvas.winfo_height() // 4)

    # The image should have been dragged
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.01)

    # Dragging the image to the right border
    for _ in range(20):
      self._config._img_canvas.event_generate(
          '<ButtonPress-3>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2)
      self._config._img_canvas.event_generate(
          '<B3-Motion>', when="now",
          x=self._config._img_canvas.winfo_width() // 4,
          y=self._config._img_canvas.winfo_height() // 2)

    # The image should have hit the border
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           1.0 - (level_max - level_min), delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           1.0, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max, delta=0.01)

    # Dragging the image to the bottom border
    for _ in range(20):
      self._config._img_canvas.event_generate(
          '<ButtonPress-3>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2)
      self._config._img_canvas.event_generate(
          '<B3-Motion>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 4)

    # The image should have hit the border
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           1.0 - (level_max - level_min), delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           1.0, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           1.0 - (level_max - level_min), delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           1.0, delta=0.01)

    # Dragging the image to the left border
    for _ in range(20):
      self._config._img_canvas.event_generate(
          '<ButtonPress-3>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2)
      self._config._img_canvas.event_generate(
          '<B3-Motion>', when="now",
          x=3 * self._config._img_canvas.winfo_width() // 4,
          y=self._config._img_canvas.winfo_height() // 2)

    # The image should have hit the border
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           0.0, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max - level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           1.0 - (level_max - level_min), delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           1.0, delta=0.01)

    # Dragging the image to the top border
    for _ in range(20):
      self._config._img_canvas.event_generate(
          '<ButtonPress-3>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=self._config._img_canvas.winfo_height() // 2)
      self._config._img_canvas.event_generate(
          '<B3-Motion>', when="now",
          x=self._config._img_canvas.winfo_width() // 2,
          y=3 * self._config._img_canvas.winfo_height() // 4)

    # The image should have hit the border
    self.assertAlmostEqual(self._config._zoom_values.x_low,
                           0.0, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.x_high,
                           level_max - level_min, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_low,
                           0.0, delta=0.01)
    self.assertAlmostEqual(self._config._zoom_values.y_high,
                           level_max - level_min, delta=0.01)

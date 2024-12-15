# coding: utf-8

from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestDrag(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args,
                     camera=FakeTestCameraSimple(min_val=0, max_val=255),
                     **kwargs)

  def test_drag(self) -> None:
    """"""

    # Looping once to load a first image
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

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

    # Click and drag the image on its corner
    self._config._img_canvas.event_generate('<ButtonPress-3>',
                                            when="now", x=0, y=0)
    self._config._img_canvas.event_generate('<B3-Motion>',
                                            when="now", x=50, y=50)

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

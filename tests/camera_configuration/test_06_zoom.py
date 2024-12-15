# coding: utf-8

from platform import system
from itertools import product

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestZoom(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args,
                     camera=FakeTestCameraSimple(min_val=0, max_val=255),
                     **kwargs)
  
  def test_zoom(self) -> None:
    """"""
    
    # Looping once to load a first image
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

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
    to_test = range(0, 255, 50)
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

        # Update the displayed image
        self._config._img_acq_sched()
        self._config._upd_var_sched()
        self._config._upd_sched()

        # Check that the displayed sub-image is the expected one
        min_, max_ = self._config._pil_img.getextrema()
        self.assertAlmostEqual(int((x_min / 255 * img_ratio + y_min / 255) /
                                   (img_ratio + 1) * 255), min_, delta=2)
        self.assertAlmostEqual(int((x_max / 255 * img_ratio + y_max / 255) /
                                   (img_ratio + 1) * 255), max_, delta=2)

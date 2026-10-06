# coding: utf-8

"""Tk image conversion and canvas resize integration."""

from re import fullmatch
import numpy as np

from ._fixtures import TkinterConfigTestCase, FakeTestCameraSimple


class TestImageRendering(TkinterConfigTestCase):
  """Check PIL modes and pixels at the Tk rendering boundary."""

  def make_camera(self) -> FakeTestCameraSimple:
    return FakeTestCameraSimple()

  def test_preview_renders_gray_and_rgb_pixels(self) -> None:
    """Tk renders the normalized core image rather than converting it again."""

    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for channels in (None, 1, 2, 3, 4):
      with self.subTest(channels=channels):
        image = gray if channels is None else \
            np.stack([gray] * channels, axis=2)
        self._config._cast_img(image)
        self._config._resize_img()

        self.assertEqual(self._config._pil_img.mode,
                         'L' if channels is None or channels < 3 else 'RGB')
        self.assertEqual(self._config._pil_img.getpixel((0, 0)),
                         0 if channels is None or channels < 3 else (0, 0, 0))


class TestResize(TkinterConfigTestCase):
  """Class for testing the behavior of the configuration window when
  resized.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def test_resize(self) -> None:
    """Tests whether the interface effectively resizes itself and the canvas
    when resizing the overall window containing it."""

    # Resizing only needs an existing histogram image, not a live worker
    self._config._hist = np.full((80, 512), 255, dtype=np.uint8)
    self.run_config_cycle()

    # Read the current sizes of the graphical objects
    hist_canvas_width = self._config._hist_canvas.winfo_width()
    hist_canvas_height = self._config._hist_canvas.winfo_height()
    hist_size_w, hist_size_h = self._config._pil_hist.size
    img_canvas_width = self._config._img_canvas.winfo_width()
    img_canvas_height = self._config._img_canvas.winfo_height()
    img_size_w, img_size_h = self._config._pil_img.size

    # Read the current geometry and extend it
    w, h, x, y = map(int, fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)',
                                    self._config.winfo_geometry()).groups())
    self._config.geometry(f"{int(2.5 * w)}x{int(1.5 * h)}+{x}+{y}")
    self._config.update_idletasks()
    self._config.update()

    # Both canvas callbacks must resize their existing image immediately,
    # without waiting for another camera acquisition.
    self.assertGreater(self._config._pil_img.size[0], img_size_w)
    self.assertGreater(self._config._pil_hist.size[0], hist_size_w)

    # Call new loops to apply the changes
    for _ in range(2):
      self.run_config_cycle()

    # All dimensions should be greater
    self.assertGreater(self._config._hist_canvas.winfo_width(),
                       hist_canvas_width)
    self.assertGreater(self._config._hist_canvas.winfo_height(),
                       hist_canvas_height)
    self.assertGreater(self._config._pil_hist.size[0], hist_size_w)
    self.assertGreater(self._config._pil_hist.size[1], hist_size_h)
    self.assertGreater(self._config._img_canvas.winfo_width(),
                       img_canvas_width)
    self.assertGreater(self._config._img_canvas.winfo_height(),
                       img_canvas_height)
    self.assertGreater(self._config._pil_img.size[0], img_size_w)
    self.assertGreater(self._config._pil_img.size[1], img_size_h)

    # Read the current sizes of the graphical objects
    hist_canvas_width = self._config._hist_canvas.winfo_width()
    hist_canvas_height = self._config._hist_canvas.winfo_height()
    hist_size_w, hist_size_h = self._config._pil_hist.size
    img_canvas_width = self._config._img_canvas.winfo_width()
    img_canvas_height = self._config._img_canvas.winfo_height()
    img_size_w, img_size_h = self._config._pil_img.size

    # Read the current geometry and reduce it
    w, h, x, y = map(int, fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)',
                                    self._config.winfo_geometry()).groups())
    self._config.geometry(f"{int(0.6 * w)}x{int(0.8 * h)}+{x}+{y}")
    self._config.update_idletasks()
    self._config.update()

    # Call new loops to apply the changes
    for _ in range(2):
      self.run_config_cycle()

    # All dimensions should be smaller
    self.assertGreater(hist_canvas_width,
                       self._config._hist_canvas.winfo_width())
    self.assertGreater(hist_canvas_height,
                       self._config._hist_canvas.winfo_height())
    self.assertGreater(hist_size_w, self._config._pil_hist.size[0])
    self.assertGreater(hist_size_h, self._config._pil_hist.size[1])
    self.assertGreater(img_canvas_width,
                       self._config._img_canvas.winfo_width())
    self.assertGreater(img_canvas_height,
                       self._config._img_canvas.winfo_height())
    self.assertGreater(img_size_w, self._config._pil_img.size[0])
    self.assertGreater(img_size_h, self._config._pil_img.size[1])

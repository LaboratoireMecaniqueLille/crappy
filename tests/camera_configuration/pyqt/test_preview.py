# coding: utf-8

"""Qt preview rendering and immediate resize behavior."""

from unittest.mock import Mock, patch
import numpy as np

from ._fixtures import PyQtConfigTestCase
from crappy.tool.camera_config.pyqt import camera_config as pyqt_config


class TestPreview(PyQtConfigTestCase):
  def test_image_preview_reports_shape_and_dtype(self) -> None:
    """Qt displays the acquired image and exposes its format to the Block."""

    config = self.make_config()
    config.show()
    config._qt_app.processEvents()
    config._update_img()

    self.assertEqual(config.shape, (100, 100))
    self.assertEqual(config.dtype, 'uint8')
    self.assertGreater(config._display_geometry.image_width, 0)
    self.assertIsNotNone(config._img_canvas.pixmap())
    self.assertEqual(config._min_max_label.text(), 'min: 0, max: 255')
    self.assertEqual(config._bits_label.text(), 'Detected bits: 8')

  def test_preview_reports_transformed_image_format(self) -> None:
    """The Qt acquisition path renders transformed images and reports them."""

    config = self.make_config()
    config._transform = lambda image: image[:20, :30].astype('uint16')
    config.show()
    config._qt_app.processEvents()
    config._update_img()

    self.assertEqual(config.shape, (20, 30))
    self.assertEqual(config.dtype, 'uint16')
    self.assertIsNotNone(config._img_canvas.pixmap())

  def test_resize_renders_existing_image_and_histogram(self) -> None:
    """A resize redraws existing content without requiring a new acquisition."""

    config = self.make_config()
    config.show()
    app = config._qt_app
    app.processEvents()
    config._update_img()
    config._hist = np.full((80, 512), 255, dtype=np.uint8)
    config._render_histogram()
    old_image = config._img_canvas.pixmap().size()
    old_histogram = config._hist_canvas.pixmap().size()

    with patch.object(self.camera, 'get_image') as acquire:
      config.resize(config.width() + 200, config.height() + 150)
      app.processEvents()

    acquire.assert_not_called()
    self.assertGreater(config._img_canvas.pixmap().height(), old_image.height())
    self.assertGreater(config._hist_canvas.pixmap().width(), old_histogram.width())
    self.assertEqual(config._hist_canvas.pixmap().size(),
                     config._hist_canvas.contentsRect().size())

  def test_acquisition_schedules_deadline_instead_of_polling(self) -> None:
    """Qt's single-shot timer waits until the next limited frame is due."""

    config = self.make_config()
    config._max_freq = 20
    with (patch.object(pyqt_config, 'monotonic', return_value=100.0),
          patch.object(config, '_update_img') as acquire,
          patch.object(config, '_sync_setting_controls') as sync,
          patch.object(config, '_acquisition_timer', Mock()) as timer):
      config._acquire_and_render()
      config._acquire_and_render()

    acquire.assert_called_once_with()
    sync.assert_called_once_with()
    self.assertEqual(config._next_acq_t, 100.05)
    self.assertEqual([call.args[0] for call in timer.start.call_args_list], [50, 50])

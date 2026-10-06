# coding: utf-8

# [custom-camera-configuration-start]
import logging

import crappy
from crappy.camera.meta_camera.camera_setting import CameraBoolSetting
from crappy.tool.camera_config.base import CameraConfig
from crappy.tool.camera_config.tkinter import TkinterCameraConfig
from crappy.tool.camera_config.pyqt import PyQtCameraConfig


class ConfirmedConfig(CameraConfig):
  """Require an explicit confirmation before accepting camera setup."""

  def _create_local_settings(self):
    self._confirmation = CameraBoolSetting('Confirm setup', default=False)
    return *super()._create_local_settings(), self._confirmation

  def _validate_close(self):
    reason = super()._validate_close()
    if reason is not None:
      return reason
    if not self._confirmation.value:
      return 'Check Confirm setup and apply the settings before closing.'
    return None

  def _on_valid_close(self):
    super()._on_valid_close()
    self.log(logging.INFO, 'Camera setup confirmed')


class ConfirmedTkConfig(ConfirmedConfig, TkinterCameraConfig):
  pass


class ConfirmedQtConfig(ConfirmedConfig, PyQtCameraConfig):
  pass


class ConfirmedCameraSource(crappy.blocks.vision.CameraSource):
  configurator = {'tkinter': ConfirmedTkConfig, 'pyqt': ConfirmedQtConfig}


def main() -> None:
  # The default window uses PyQt6; add config_backend='tkinter' to compare
  source = ConfirmedCameraSource(
      'FakeCamera',
      width=160, height=120, speed=25, fps=10, freq=20)
  display = crappy.blocks.vision.ImageDisplayer(
      title='Confirmed Camera setup', backend='mpl', framerate=10)
  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.img_link(source, display)
  crappy.link(source, stop)
  crappy.start()


if __name__ == '__main__':
  main()
# [custom-camera-configuration-end]

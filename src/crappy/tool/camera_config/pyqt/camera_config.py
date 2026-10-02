# coding: utf-8

"""PyQt6 camera configurator using the toolkit-independent configuration core.

PyQt6 remains optional until one of these configurators is instantiated.
"""

from __future__ import annotations
from abc import ABCMeta
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from math import ceil
from multiprocessing import Event, Queue, synchronize
from multiprocessing.queues import Queue as MPQueue
from queue import Empty
from time import monotonic, time
from typing import Any
import logging
import os
import numpy as np

from ..base import CameraConfig, CameraConfigBoxes
from ..base._configuration_lifecycle import ConfigurationLifecycle
from ..config_tools import HistogramProcess
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import (CameraBoolSetting,
                                                  CameraChoiceSetting,
                                                  CameraScaleSetting,
                                                  CameraSetting)
from ...._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule('pillow')

try:
  from PyQt6.QtCore import QEvent, QEventLoop, Qt, QTimer
  from PyQt6.QtGui import QCloseEvent, QImage, QPalette, QPixmap
  from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)
except (ModuleNotFoundError, ImportError):
  _missing_pyqt = OptionalModule('PyQt6')
  QEvent = QEventLoop = Qt = QTimer = _missing_pyqt
  QCloseEvent = QImage = QPalette = QPixmap = _missing_pyqt
  QApplication = QButtonGroup = QCheckBox = QFrame = QGroupBox = _missing_pyqt
  QHBoxLayout = QLabel = QMessageBox = QPushButton = _missing_pyqt
  QRadioButton = QScrollArea = QSizePolicy = _missing_pyqt
  QSlider = QVBoxLayout = _missing_pyqt
  # A base class must be a type, even when Qt is unavailable
  QWidget = object


@dataclass
class _QtSettingControl:
  """Qt editor state for one setting, owned by the configuration window.

  Attributes:
    widget: The checkbox, scale container, or choice group shown to the user.
    revision: The setting revision last copied into the control. It is not a
      count of user edits, so an unchanged revision preserves pending edits.
    slider: The scale widget, if this is a scale setting.
    value_label: The label displaying a scale's requested value.
    buttons: The radio button group for a choice setting.
    choices_layout: The layout holding the choice buttons, needed when the
      available choices change after a setting is applied.
  """

  widget: QWidget
  revision: int
  slider: QSlider | None = None
  value_label: QLabel | None = None
  buttons: QButtonGroup | None = None
  choices_layout: QVBoxLayout | None = None


class _QtABCMeta(ABCMeta, type(QWidget)):
  """Combine the abstract core's metaclass with Qt's widget metaclass."""


class PyQtCameraConfig(CameraConfig, QWidget, metaclass=_QtABCMeta):
  """PyQt6 window for previewing images and tuning Camera settings.

  Like :class:`~crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig`, this window displays
  the image, its pixel histogram, acquisition information, and the available
  Camera settings. The mousewheel zooms the image and right-dragging pans it.
  Specialized camera Blocks can add selection gestures and action buttons.

  Toolkit-independent state and image interactions are inherited from
  :class:`~crappy.tool.camera_config.base.camera_config.\
CameraConfig`. This class owns the Qt widgets, timers, and rendering.
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               *_, **__) -> None:
    """Initializes the window and its histogram calculation process.

    Args:
      camera: The Camera object in charge of acquiring the images.
      log_queue: The queue forwarding log messages to the main process.
      log_level: The minimum logging level of the Crappy script.
      max_freq: The maximum frequency at which the preview may acquire images.
      transform: An optional callable applied to images before previewing them
        and reporting their shape and data type to the owning Block.
    """

    # Reuse an application created by the caller, or create one for this window
    self._qt_app = self._get_application()
    self._window_closed = False
    self._shutdown_requested: Callable[[], bool] | None = None
    self._event_loop: QEventLoop | None = None
    self._last_upd_t: float | None = None
    self._next_acq_t = -float('inf')
    self._hist: np.ndarray | None = None
    self._setting_controls: dict[CameraSetting, _QtSettingControl] = {}

    # Abort early if the Camera settings cannot be initialized
    try:
      super().__init__(camera, log_queue, log_level, max_freq, transform)
    except BaseException:
      self._window_closed = True
      try:
        self.close()
      except Exception:
        pass
      raise

    self._stop_event: synchronize.Event = Event()
    self._processing_event: synchronize.Event = Event()
    # A constructor failure must close any queues that were already created
    created_queues: list[MPQueue] = []
    try:
      self._img_in: MPQueue = Queue(maxsize=0)
      created_queues.append(self._img_in)
      self._img_out: MPQueue = Queue(maxsize=0)
      created_queues.append(self._img_out)
      self._histogram_process = HistogramProcess(
          stop_event=self._stop_event,
          processing_event=self._processing_event,
          img_in=self._img_in,
          img_out=self._img_out,
          log_level=self._log_level,
          log_queue=self._log_queue)
      self._lifecycle = ConfigurationLifecycle(
          self._stop_event, self._histogram_process,
          (self._img_in, self._img_out), self.log)
    except BaseException:
      for queue in created_queues:
        try:
          queue.cancel_join_thread()
        except Exception as error:
          self.log(logging.ERROR, 'Could not join histogram queue thread',
                   exc_info=(type(error), error, error.__traceback__))
        try:
          queue.close()
        except Exception as error:
          self.log(logging.ERROR, 'Could not close histogram queue',
                   exc_info=(type(error), error, error.__traceback__))
      self._window_closed = True
      self.close()
      raise

    # Separate timers acquire images, refresh the FPS, and watch Block shutdown
    self._acquisition_timer = QTimer(self)
    self._acquisition_timer.setSingleShot(True)
    self._acquisition_timer.timeout.connect(self._guard(self._acquire_and_render))
    self._indicator_timer = QTimer(self)
    self._indicator_timer.setInterval(500)
    self._indicator_timer.timeout.connect(self._guard(self._update_indicators))
    self._shutdown_timer = QTimer(self)
    self._shutdown_timer.setInterval(25)
    self._shutdown_timer.timeout.connect(self._guard(self._check_shutdown))

    # Assemble the interface only after the core and histogram process exist
    try:
      self.setWindowTitle(f'Configuration window for the camera: '
                          f'{type(camera).__name__}')
      self._set_layout()
      self._add_settings()
    except BaseException:
      self.stop()
      raise

  @staticmethod
  def _get_application() -> QApplication:
    """Returns the existing Qt application or creates one for this window.

    OpenCV may put its own Qt plugins on ``QT_QPA_PLATFORM_PLUGIN_PATH``. That
    path can prevent PyQt6 from loading its platform plugin, so it is excluded
    only while constructing the application and then restored.
    """

    app = QApplication.instance()
    if app is not None:
      if not isinstance(app, QApplication):
        raise RuntimeError("A non-widget Qt application already exists")
      return app

    # Leave unrelated Qt plugin locations available to the application
    original = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')
    if original is not None:
      paths = original.split(os.pathsep)
      filtered = [path for path in paths if
                  tuple(os.path.normpath(path).split(os.sep)[-3:]) !=
                  ('cv2', 'qt', 'plugins')]
      if filtered != paths:
        if filtered:
          os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(filtered)
        else:
          os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

    try:
      return QApplication([])
    finally:
      if original is not None:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = original

  def _guard(self, callback: Callable[..., Any]) -> Callable[..., None]:
    """Wraps a Qt callback so failures can be raised after the event loop.

    Args:
      callback: The function connected to a Qt signal.
    """

    def guarded(*args: Any) -> None:
      try:
        callback(*args)
      except BaseException as error:
        self._record_callback_failure(error)

    return guarded

  def _record_callback_failure(self, error: BaseException) -> None:
    """Reports a callback failure and asks the window to close.

    The lifecycle retains the original traceback for :meth:`run` to raise
    after the Qt event loop has stopped. Keyboard interrupts close silently.
    """

    self._lifecycle.record_callback_failure(error, error.__traceback__)
    try:
      if not isinstance(error, KeyboardInterrupt):
        QMessageBox.critical(self, 'Error!',
                             f'{type(error).__name__}\n{error}')
    except Exception as dialog_error:
      self.log(logging.ERROR, 'Could not display configuration error',
               exc_info=(type(dialog_error), dialog_error,
                         dialog_error.__traceback__))
    finally:
      self._lifecycle.request_close(self.stop)

  def run(self) -> None:
    """Runs the configuration window until it closes.

    The Qt event loop hides exceptions raised by signal callbacks, so those
    failures are saved by :meth:`_guard` and raised after the loop exits.
    Histogram resources are also released if opening the window fails.
    """

    try:
      self._lifecycle.raise_if_failed()
      if self._window_closed:
        raise RuntimeError('Cannot run a closed configuration window')
      self._check_shutdown()
      if not self._window_closed:
        self._event_loop = QEventLoop(self)
        self.show()
        self.start()
        self._event_loop.exec()
    except BaseException as error:
      try:
        self.stop()
      except Exception as cleanup_error:
        self.log(logging.ERROR, 'Could not clean up configuration window',
                 exc_info=(type(cleanup_error), cleanup_error,
                           cleanup_error.__traceback__))
      if not isinstance(error, KeyboardInterrupt):
        self._lifecycle.raise_if_failed()
      raise

    self.stop()
    self._lifecycle.raise_if_failed()

  def start(self) -> None:
    """Starts histogram processing and the periodic GUI updates.

    Image acquisition uses a single-shot timer to respect ``max_freq``. A
    separate timer updates the displayed FPS twice per second.
    """

    if self._lifecycle.closed:
      raise RuntimeError('Cannot start a closed configuration window')
    # Start the histogram worker before the first image is acquired
    self._histogram_process.start()
    self._lifecycle.mark_histogram_started()
    self._n_loops = 0
    self._last_upd_t = time()
    self._next_acq_t = -float('inf')
    self._acquisition_timer.start(0)
    self._indicator_timer.start()
    if self._shutdown_requested is not None:
      self._shutdown_timer.start()

  def watch_shutdown(self, requested: Callable[[], bool]) -> None:
    """Closes the window when the owning Block requests shutdown.

    Args:
      requested: A callback returning whether the Block is stopping or its
        preparation has failed.
    """

    self._shutdown_requested = requested
    if self._event_loop is not None and not self._window_closed:
      self._shutdown_timer.start()

  def _check_shutdown(self) -> None:
    """Checks the Block shutdown request from the Qt event loop."""

    if (not self._window_closed and self._shutdown_requested is not None and
        self._shutdown_requested()):
      self.stop()

  def finish(self) -> None:
    """Validates a user close and finalizes the selected configuration.

    A Block shutdown bypasses validation, while an invalid user selection
    keeps the window open so it can be corrected.
    """

    # The Block may need to close even when its current selection is invalid
    if self._shutdown_requested is not None and self._shutdown_requested():
      self.stop()
      return
    if (reason := self._validate_close()) is not None:
      self.log(logging.WARNING, reason)
      QMessageBox.critical(self, 'Error !', reason)
      return
    self._on_valid_close()
    self.stop()

  def closeEvent(self, event: QCloseEvent) -> None:
    """Routes window-manager close requests through :meth:`finish`."""

    if self._window_closed:
      event.accept()
      return
    try:
      self.finish()
    except BaseException as error:
      self._record_callback_failure(error)
    if self._window_closed:
      event.accept()
    else:
      event.ignore()

  def stop(self) -> None:
    """Closes the window without validation and releases its resources.

    This is also called during shutdown or after a callback failure, so it
    remains safe to call more than once.
    """

    if self._window_closed:
      self._lifecycle.close_resources()
      return
    self._window_closed = True
    self._acquisition_timer.stop()
    self._indicator_timer.stop()
    self._shutdown_timer.stop()
    try:
      self.close()
    finally:
      if self._event_loop is not None:
        self._event_loop.quit()
      self._lifecycle.close_resources()

  def _set_frame_style(self) -> None:
    """Rounds frame outlines and softens their current theme's text color."""

    # Soften the frame outlines while keeping them in the current Qt palette
    self._frame_palette = self._qt_app.palette()
    text = self._frame_palette.color(QPalette.ColorRole.WindowText)
    # Clear cached widget palettes before applying a new theme's outlines
    self.setStyleSheet('')
    self.setStyleSheet(f'''
        QFrame[configFrame="true"], QGroupBox {{
          border: 1px solid rgba({text.red()}, {text.green()}, {text.blue()}, 80);
          border-radius: 6px;
        }}
        QLabel[configFrame="true"], QScrollArea[configFrame="true"] {{
          padding: 4px;
        }}
        QGroupBox {{
          margin-top: 0.5em;
        }}
        QGroupBox::title {{
          subcontrol-origin: margin;
          left: 8px;
          padding: 0 3px;
        }}
    ''')

  def _set_layout(self) -> None:
    """Places the histogram, image, information, and settings on the window.

    The layout follows the Tk configurator: the histogram sits above the
    image, with indicators and setting controls in a panel on the right.
    """

    self._set_frame_style()
    # Use the same minimum image area as Tk for large and small screens
    screen = self._qt_app.primaryScreen()
    size = screen.availableGeometry() if screen is not None else None
    if size is not None and size.width() >= 1600 and size.height() >= 900:
      min_width, min_height = 800, 600
    else:
      min_width, min_height = 600, 450

    # The left side holds the histogram and the camera image
    main = QHBoxLayout(self)
    graphical = QVBoxLayout()
    main.addLayout(graphical, stretch=1)

    self._hist_canvas = QLabel()
    self._hist_canvas.setMinimumSize(min_width, 80)
    self._hist_canvas.setFrameShape(QFrame.Shape.Box)
    self._hist_canvas.setProperty('configFrame', True)
    self._hist_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self._hist_canvas.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Fixed)
    self._hist_canvas.setBackgroundRole(QPalette.ColorRole.Base)
    self._hist_canvas.setAutoFillBackground(True)
    self._hist_canvas.installEventFilter(self)
    graphical.addWidget(self._hist_canvas)

    # Keep the image's aspect ratio when it is rendered into this canvas
    self._img_canvas = QLabel()
    self._img_canvas.setMinimumSize(min_width, min_height)
    self._img_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self._img_canvas.setSizePolicy(QSizePolicy.Policy.Ignored,
                                   QSizePolicy.Policy.Expanding)
    self._img_canvas.setBackgroundRole(QPalette.ColorRole.Base)
    self._img_canvas.setAutoFillBackground(True)
    self._img_canvas.setMouseTracking(True)
    self._img_canvas.installEventFilter(self)
    graphical.addWidget(self._img_canvas, stretch=1)

    # The right side contains live information, actions, and camera settings
    side = QFrame()
    side.setFrameShape(QFrame.Shape.Box)
    side.setProperty('configFrame', True)
    side.setMinimumWidth(300)
    side.setMaximumWidth(360)
    main.addWidget(side)
    side_layout = QVBoxLayout(side)

    # Display the same acquisition and cursor indicators as the Tk window
    info = QFrame()
    info.setFrameShape(QFrame.Shape.Box)
    info.setProperty('configFrame', True)
    info_layout = QVBoxLayout(info)
    side_layout.addWidget(info)
    self._fps_label = QLabel()
    self._fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    info_layout.addWidget(self._fps_label)
    self._auto_range_button = QCheckBox('Auto range')
    self._auto_range_button.clicked.connect(self._guard(self._on_auto_range))
    info_layout.addWidget(self._auto_range_button,
                          alignment=Qt.AlignmentFlag.AlignHCenter)
    self._auto_apply_button = QCheckBox('Auto apply')
    self._auto_apply_button.clicked.connect(self._guard(self._on_auto_apply))
    info_layout.addWidget(self._auto_apply_button,
                          alignment=Qt.AlignmentFlag.AlignHCenter)
    self._min_max_label = QLabel()
    self._bits_label = QLabel()
    self._zoom_label = QLabel()
    self._reticle_label = QLabel()
    for label in (self._min_max_label, self._bits_label, self._zoom_label,
                  self._reticle_label):
      label.setAlignment(Qt.AlignmentFlag.AlignCenter)
      info_layout.addWidget(label)

    # Remind the user that closing the window validates the configuration
    self._validate_text = QLabel(
        'To validate the choice of the settings and start the test, simply '
        'close this window.')
    self._validate_text.setWordWrap(True)
    palette = self._validate_text.palette()
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.red)
    self._validate_text.setPalette(palette)
    side_layout.addWidget(self._validate_text)

    # Add the standard Apply button followed by specialized Block actions
    self._apply_button = QPushButton('Apply Settings')
    self._apply_button.clicked.connect(
        self._guard(lambda _checked=False: self._update_settings()))
    side_layout.addWidget(self._apply_button)
    self._action_buttons: dict[str, QPushButton] = {}
    for action in self._extra_actions():
      if action.id in self._action_buttons:
        raise ValueError(f'Duplicate configuration action: {action.id}')
      button = QPushButton(action.label)
      button.clicked.connect(
          self._guard(lambda _checked=False, callback=action.callback:
                      callback()))
      side_layout.addWidget(button)
      self._action_buttons[action.id] = button

    # Scroll the settings independently of the image and information panel
    settings = QScrollArea()
    settings.setWidgetResizable(True)
    settings.setFrameShape(QFrame.Shape.Box)
    settings.setProperty('configFrame', True)
    self._settings_widget = QWidget()
    self._settings_layout = QVBoxLayout(self._settings_widget)
    self._settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    settings.setWidget(self._settings_widget)
    side_layout.addWidget(settings, stretch=1)
    self._sync_indicators()

  def _add_settings(self) -> None:
    """Creates controls for local settings, then for Camera settings.

    Camera settings are sorted by type to match the Tk interface.
    """

    for setting in self._setting_manager.local_settings:
      self._add_setting_control(setting)
    for setting in sorted(self._camera.settings.values(),
                          key=lambda item: item.type.__name__):
      self._add_setting_control(setting)

  def _add_setting_control(self, setting: CameraSetting) -> None:
    """Adds the appropriate Qt editor for one Camera setting.

    Args:
      setting: The boolean, scale, or choice setting to display.
    """

    # Boolean settings use a checkbox
    if isinstance(setting, CameraBoolSetting):
      checkbox = QCheckBox(setting.name)
      checkbox.setChecked(bool(setting.value))
      checkbox.clicked.connect(self._guard(self._auto_apply_settings))
      self._settings_layout.addWidget(checkbox)
      control = _QtSettingControl(checkbox, setting.revision)
    # Scale settings show their current request next to a horizontal slider
    elif isinstance(setting, CameraScaleSetting):
      frame = QWidget()
      layout = QVBoxLayout(frame)
      label = QLabel()
      slider = QSlider(Qt.Orientation.Horizontal)
      self._configure_slider(setting, slider)
      slider.setValue(self._slider_position(setting, slider, setting.value))
      label.setText(self._scale_label(setting,
                                      self._slider_value(setting, slider)))
      slider.valueChanged.connect(
          lambda _value, s=setting, w=slider, l=label: l.setText(
              self._scale_label(s, self._slider_value(s, w))))
      slider.sliderReleased.connect(self._guard(self._auto_apply_settings))
      layout.addWidget(label)
      layout.addWidget(slider)
      self._settings_layout.addWidget(frame)
      control = _QtSettingControl(frame, setting.revision, slider, label)
    # Choice settings use a group of mutually exclusive radio buttons
    elif isinstance(setting, CameraChoiceSetting):
      group = QGroupBox(setting.name)
      layout = QVBoxLayout(group)
      buttons = QButtonGroup(group)
      self._set_choices(setting, buttons, layout)
      self._settings_layout.addWidget(group)
      control = _QtSettingControl(group, setting.revision,
                                  buttons=buttons, choices_layout=layout)
    else:
      return

    # Only registered controls need to be synchronized with model revisions
    self._setting_manager.register_local(setting)
    self._setting_controls[setting] = control

  def _set_choices(self, setting: CameraChoiceSetting,
                   buttons: QButtonGroup, layout: QVBoxLayout) -> None:
    """Rebuilds the radio buttons when a setting's choices change.

    Args:
      setting: The choice setting being displayed.
      buttons: The group containing its radio buttons.
      layout: The layout in which the buttons appear.
    """

    # Reloading a Camera setting may replace its entire choice list
    for button in buttons.buttons():
      buttons.removeButton(button)
      layout.removeWidget(button)
      button.deleteLater()
    for choice in setting.choices:
      button = QRadioButton(str(choice))
      button.setProperty('choice', choice)
      buttons.addButton(button)
      layout.addWidget(button)
      button.clicked.connect(self._guard(self._auto_apply_settings))
      if choice == setting.value:
        button.setChecked(True)

  @staticmethod
  def _configure_slider(setting: CameraScaleSetting, slider: QSlider) -> None:
    """Maps a Camera scale onto the integer range accepted by Qt sliders."""

    span = setting.highest - setting.lowest
    step = setting.step or (1 if setting.type is int else span / 1000)
    slider.setRange(0, min(max(round(span / step), 1), 2_000_000_000))

  @staticmethod
  def _slider_position(setting: CameraScaleSetting, slider: QSlider,
                       value: int | float) -> int:
    """Returns the slider position corresponding to a setting value."""

    span = setting.highest - setting.lowest
    return round((value - setting.lowest) * slider.maximum() / span)

  @staticmethod
  def _slider_value(setting: CameraScaleSetting, slider: QSlider) -> int | float:
    """Returns the requested Camera value at the current slider position."""

    span = setting.highest - setting.lowest
    step = setting.step or (1 if setting.type is int else span / 1000)
    value = setting.lowest + slider.value() * step
    return setting.type(min(value, setting.highest))

  @staticmethod
  def _scale_label(setting: CameraScaleSetting, value: int | float) -> str:
    """Formats a scale's label without changing the requested Camera value.

    Args:
      setting: The scale setting whose name and precision are displayed.
      value: The requested or effective value to display.
    """

    if setting.type is int:
      return f'{setting.name} : {value}'
    # Keep fractional steps and offsets, without showing binary float noise
    step = setting.step or (setting.highest - setting.lowest) / 1000
    decimals = max(0, -min(Decimal(str(number)).normalize().as_tuple().exponent
                          for number in (step, setting.lowest)))
    return f'{setting.name} : {round(value, decimals):.10g}'

  def _sync_setting_controls(self,
                             settings: Iterable[CameraSetting] | None = None
                             ) -> None:
    """Copies changed setting values back into their Qt controls.

    A control is left untouched when its model revision has not changed, so a
    pending user edit is not overwritten by the preview loop.

    Args:
      settings: The settings to inspect, or all displayed settings if omitted.
    """

    for setting in (self._setting_controls if settings is None else settings):
      control = self._setting_controls.get(setting)
      if control is None or control.revision == setting.revision:
        continue
      if isinstance(setting, CameraBoolSetting):
        control.widget.setChecked(bool(setting.value))
      elif isinstance(setting, CameraScaleSetting):
        assert control.slider is not None and control.value_label is not None
        control.slider.blockSignals(True)
        self._configure_slider(setting, control.slider)
        control.slider.setValue(self._slider_position(
            setting, control.slider, setting.value))
        control.slider.blockSignals(False)
        control.value_label.setText(self._scale_label(setting, setting.value))
      elif isinstance(setting, CameraChoiceSetting):
        assert control.buttons is not None and control.choices_layout is not None
        self._set_choices(setting, control.buttons, control.choices_layout)
      control.revision = setting.revision

  def _requested_value(self, setting: CameraSetting,
                       control: _QtSettingControl) -> Any:
    """Reads the value currently requested in a setting's Qt control."""

    if isinstance(setting, CameraBoolSetting):
      return control.widget.isChecked()
    if isinstance(setting, CameraScaleSetting):
      assert control.slider is not None
      return self._slider_value(setting, control.slider)
    if isinstance(setting, CameraChoiceSetting):
      assert control.buttons is not None
      button = control.buttons.checkedButton()
      return button.property('choice') if button is not None else setting.value
    return setting.value

  def _update_settings(self) -> None:
    """Applies user-requested settings and refreshes dependent controls.

    Settings are processed in the manager's order. Before each one, model
    changes made by an earlier setting are copied into the corresponding
    controls, as a Camera setter may alter other settings or their choices.
    """

    for setting in self._setting_manager.settings:
      self._sync_setting_controls()
      control = self._setting_controls.get(setting)
      if control is None:
        continue
      result = self._setting_manager.apply(
          setting, self._requested_value(setting, control))
      self._sync_setting_controls(result.changed)

  def _auto_apply_settings(self, *_: Any) -> None:
    """Applies a control change if Auto apply is enabled."""

    if self._display_state.auto_apply:
      self._update_settings()

  def _on_auto_range(self, checked: bool) -> None:
    """Updates whether the histogram adjusts the preview's pixel range."""

    self._display_state.auto_range = checked

  def _on_auto_apply(self, checked: bool) -> None:
    """Enables automatic setting updates and disables the Apply button."""

    self._display_state.auto_apply = checked
    self._apply_button.setEnabled(not checked)

  def _sync_indicators(self) -> None:
    """Copies the current FPS, pixel, zoom, and cursor values into labels."""

    state = self._display_state
    self._fps_label.setText(f'fps = {state.fps:.2f}\n'
                            '(might be lower in this GUI than actual)')
    self._min_max_label.setText(f'min: {state.min_pixel:d}, '
                                f'max: {state.max_pixel:d}')
    self._bits_label.setText(f'Detected bits: {state.detected_bits:d}')
    self._zoom_label.setText(f'Zoom: {state.zoom_percent:.1f}%')
    self._reticle_label.setText(f'X: {state.reticle_x:d}, '
                                f'Y: {state.reticle_y:d}, '
                                f'V: {state.reticle_value:d}')

  def _acquire_and_render(self) -> None:
    """Acquires an image when due and schedules the next preview update.

    A single-shot timer follows the Camera Block's frequency limit without
    blocking Qt's event loop between acquisitions.
    """

    now = monotonic()
    if self._max_freq is None or now >= self._next_acq_t:
      if self._max_freq is not None:
        self._next_acq_t = now + 1 / self._max_freq
      self._update_img()
      self._sync_setting_controls()
    if not self._window_closed:
      delay = (1 if self._max_freq is None else
               max(1, ceil(1000 * (self._next_acq_t - monotonic()))))
      self._acquisition_timer.start(delay)

  def _update_indicators(self) -> None:
    """Calculates the recent preview FPS and refreshes the displayed values."""

    if self._last_upd_t is None:
      return
    elapsed = time() - self._last_upd_t
    self._display_state.fps = self._n_loops / elapsed if elapsed > 0 else 0.0
    self._n_loops = 0
    self._last_upd_t = time()
    self._sync_indicators()

  def _update_img(self) -> None:
    """Acquires and draws an image, its histogram, and its pixel indicators."""

    if not self._acquire_image():
      return
    self._read_image_geometry()
    self._draw_overlay()
    self._render_image()
    self._calc_hist()
    self._render_histogram()
    self._update_pixel_value()
    self._sync_indicators()

  def _read_image_geometry(self) -> None:
    """Updates the core with the current size of the Qt image canvas."""

    self._display_geometry.width = self._img_canvas.width()
    self._display_geometry.height = self._img_canvas.height()

  def _render_image(self) -> None:
    """Draws the visible, zoomed part of the image on the Qt canvas."""

    if self._img is None:
      return
    # Crop the image according to the current zoom before fitting the canvas
    img_height, img_width, *_ = self._img.shape
    zoom = self._zoom_values
    cropped = self._img[int(img_height * zoom.y_low):
                        int(img_height * zoom.y_high),
                        int(img_width * zoom.x_low):
                        int(img_width * zoom.x_high)]
    if not cropped.size:
      self._img_canvas.clear()
      self._display_geometry.image_width = 0
      self._display_geometry.image_height = 0
      return
    height, width, *_ = cropped.shape
    fit_width, fit_height = self._display_geometry.fit(width, height)
    if not fit_width or not fit_height:
      return
    # Qt must own a copy because the NumPy image may be replaced next frame
    cropped = np.ascontiguousarray(cropped)
    if cropped.ndim == 2:
      fmt = QImage.Format.Format_Grayscale8
    else:
      fmt = QImage.Format.Format_RGB888
    image = QImage(cropped.data, width, height, cropped.strides[0], fmt).copy()
    pixmap = QPixmap.fromImage(image).scaled(
        fit_width, fit_height, Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation)
    self._img_canvas.setPixmap(pixmap)
    self._display_geometry.image_width = fit_width
    self._display_geometry.image_height = fit_height

  def _calc_hist(self) -> None:
    """Sends a small image to the histogram process and reads its result.

    A new image is not submitted while the worker reports that it is busy.
    The most recent completed histogram is retained for rendering.
    """

    if self._original_img is None:
      return
    # The worker needs at most a 320 by 240 grayscale sample
    if not self._processing_event.is_set():
      hist_img = Image.fromarray(self._original_img)
      if hist_img.width > 320 or hist_img.height > 240:
        factor = min(320 / hist_img.width, 240 / hist_img.height)
        hist_img = hist_img.resize((max(int(hist_img.width * factor), 1),
                                    max(int(hist_img.height * factor), 1)))
      if self._original_img.ndim == 3:
        hist_img = hist_img.convert('L')
      self._img_in.put_nowait((hist_img, self._display_state.auto_range,
                               self._low_thresh, self._high_thresh))
    # Drain completed results so the display uses the newest histogram
    try:
      while True:
        self._hist = self._img_out.get_nowait()
    except Empty:
      pass

  def _render_histogram(self) -> None:
    """Draws the histogram in the active Qt light or dark palette.

    The histogram process returns a grayscale image. Its background, bars,
    and range markers are recolored here during preview updates or when the
    application theme changes.
    """

    if self._hist is None:
      return
    bounds = self._hist_canvas.contentsRect()
    if bounds.isEmpty():
      return
    # The worker encodes bars as 0 and range markers as 127
    palette = self._qt_app.palette()
    background = palette.color(QPalette.ColorRole.Base)
    foreground = palette.color(QPalette.ColorRole.Text)
    marker = palette.color(QPalette.ColorRole.Highlight)
    rgb = np.empty((*self._hist.shape, 3), dtype=np.uint8)
    rgb[:] = (background.red(), background.green(), background.blue())
    rgb[self._hist == 0] = (foreground.red(), foreground.green(),
                            foreground.blue())
    rgb[self._hist == 127] = (marker.red(), marker.green(), marker.blue())
    rgb = np.ascontiguousarray(rgb)
    # Copy the temporary NumPy buffer before creating the Qt pixmap
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, rgb.strides[0],
                   QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image).scaled(
        bounds.width(), bounds.height(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation)
    self._hist_canvas.setPixmap(pixmap)

  def _draw_overlay(self) -> None:
    """Draws the selection overlay added by a specialized camera Block."""

    ...

  def eventFilter(self, watched: QWidget, event: QEvent) -> bool:
    """Translates Qt mouse and resize events into core image interactions.

    Right-dragging pans, the mousewheel zooms, and left-dragging defines a
    selection box for the specialized configurators. Cursor motion updates
    the displayed pixel position and value.
    """

    try:
      # The histogram only needs repainting when its canvas changes size
      if watched is self._hist_canvas:
        if event.type() == QEvent.Type.Resize:
          self._render_histogram()
      elif watched is self._img_canvas:
        kind = event.type()
        if kind == QEvent.Type.Resize:
          self._read_image_geometry()
          if self._img is not None:
            self._draw_overlay()
          self._render_image()
        # Mouse events use widget coordinates shared with the display core
        elif kind == QEvent.Type.MouseMove:
          pos = event.position().toPoint()
          x, y = pos.x(), pos.y()
          if event.buttons() & Qt.MouseButton.RightButton:
            self._pan_to(x, y)
          if event.buttons() & Qt.MouseButton.LeftButton and isinstance(
              self, CameraConfigBoxes):
            self._extend_box_to(x, y)
          if self._point_at(x, y):
            self._sync_indicators()
        elif kind == QEvent.Type.MouseButtonPress:
          pos = event.position().toPoint()
          if event.button() == Qt.MouseButton.RightButton:
            self._begin_pan(pos.x(), pos.y())
          elif (event.button() == Qt.MouseButton.LeftButton and
                isinstance(self, CameraConfigBoxes)):
            self._start_box_at(pos.x(), pos.y())
        elif (kind == QEvent.Type.MouseButtonRelease and
              event.button() == Qt.MouseButton.LeftButton and
              isinstance(self, CameraConfigBoxes)):
          self._complete_box_selection()
        elif kind == QEvent.Type.Wheel:
          pos = event.position().toPoint()
          direction = (event.angleDelta().y() > 0) - (event.angleDelta().y() < 0)
          if self._zoom_at(pos.x(), pos.y(), direction):
            self._read_image_geometry()
            if self._img is not None:
              self._draw_overlay()
            self._render_image()
            self._sync_indicators()
          return True
    except BaseException as error:
      # Exceptions inside Qt event filters are not reliably propagated
      self._record_callback_failure(error)
      return True
    return QWidget.eventFilter(self, watched, event)

  def changeEvent(self, event: QEvent) -> None:
    """Recolors the frame outlines and histogram when the palette changes."""

    QWidget.changeEvent(self, event)
    if event.type() in (QEvent.Type.PaletteChange,
                        QEvent.Type.ApplicationPaletteChange):
      if hasattr(self, '_hist_canvas'):
        if self._frame_palette != self._qt_app.palette():
          self._set_frame_style()
        self._render_histogram()

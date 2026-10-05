# coding: utf-8

from __future__ import annotations
import tkinter as tk
from tkinter.messagebox import showerror
from platform import system
from math import ceil
import numpy as np
from time import monotonic, time
import logging
import locale
from dataclasses import dataclass
from multiprocessing import Event, Queue, synchronize
from multiprocessing.queues import Queue as MPQueue
from queue import Empty
from types import TracebackType
from typing import TYPE_CHECKING
from collections.abc import Callable, Iterable

from ..base import CameraConfig, ConfigurationLifecycle
from ..config_tools import HistogramProcess
from ....camera.meta_camera.camera_setting import (
  CameraSetting, CameraBoolSetting, CameraChoiceSetting, CameraScaleSetting)
from ....camera.meta_camera import Camera
from ...._global import OptionalModule

try:
  from PIL import ImageTk, Image
except (ModuleNotFoundError, ImportError):
  ImageTk = OptionalModule("pillow")
  Image = OptionalModule("pillow")

if TYPE_CHECKING:
  from PIL import Image, ImageTk


@dataclass
class _TkSettingControl:
  """Tk editor state for one setting, owned by the configuration window.

  Attributes:
    variable: The value currently requested by the user in the Tk control.
      It may differ from the effective ``setting.value`` until Apply.
    widget: The checkbutton or scale, or the radio buttons for a choice.
    revision: The setting revision last copied into this control. This is not
      a count of user edits, comparing it with ``setting.revision`` lets the
      view avoid overwriting an unrelated pending edit.
    frame: The parent frame for choice radio buttons, needed when ``reload()``
      changes the number of choices. :obj:`None` for other setting types.
  """

  variable: tk.Variable
  widget: tk.Widget | list[tk.Radiobutton]
  revision: int
  frame: tk.Frame | None = None


class TkinterCameraConfig(CameraConfig, tk.Tk):
  """Tkinter window for previewing
  :class:`~crappy.camera.meta_camera.camera.Camera` images and adjusting
  settings.

  The window shows an image, pixel histogram, preview frames per second (FPS),
  pixel-value indicators, and :class:`~crappy.camera.meta_camera.camera.Camera`
  settings. The mouse wheel zooms, and a right-button drag pans. Auto range
  changes preview contrast only. Apply Settings writes pending edits, while
  Auto apply writes them after a control change or slider release. Closing does
  not apply unsubmitted edits.

  The shared
  :class:`~crappy.tool.camera_config.base.camera_config.CameraConfig` owns
  image and setting models. This backend owns widgets, event scheduling,
  rendering, and histogram-worker resources. Use
  :meth:`run() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.run>` to configure synchronously. A user close validates
  and finalizes the selection, while
  :meth:`stop() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.stop>` and :class:`~crappy.blocks.meta_block.block.Block`
  shutdown bypass validation.

  Requires Tk support from the Python installation and Pillow.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Camera_config* to *CameraConfig*
  .. versionchanged:: 2.1.0 renamed from *CameraConfig* to
     *TkinterCameraConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               *_,
               **__) -> None:
    """Builds the window and histogram resources without starting acquisition.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.

        .. versionadded:: 2.0.0
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.

        .. versionadded:: 2.0.0
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.

        .. versionadded:: 2.0.0
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.

        .. versionadded:: 2.1.0
    """

    self._hist_width: int = 0
    self._hist_height: int = 0
    self._setting_controls: dict[CameraSetting, _TkSettingControl] = dict()

    # A Qt application created elsewhere may already have changed this locale
    locale.setlocale(locale.LC_NUMERIC, 'C')

    # Abort early in case an exception is caught while instantiating settings
    try:
      super().__init__(camera, log_queue, log_level, max_freq, transform)
    except BaseException:
      try:
        self.destroy()
      except (Exception,):
        pass
      raise

    self._window_closed: bool = False
    self._stop_event: synchronize.Event = Event()
    self._processing_event: synchronize.Event = Event()
    # A constructor failure must close queues
    created_queues: list[MPQueue] = list()
    try:
      self._img_in: MPQueue = Queue(maxsize=0)
      created_queues.append(self._img_in)
      self._img_out: MPQueue = Queue(maxsize=0)
      created_queues.append(self._img_out)
      self._histogram_process: HistogramProcess = HistogramProcess(
          stop_event=self._stop_event,
          processing_event=self._processing_event,
          img_in=self._img_in,
          img_out=self._img_out,
          log_level=self._log_level,
          log_queue=self._log_queue)
      self._lifecycle: ConfigurationLifecycle = ConfigurationLifecycle(
          self._stop_event, self._histogram_process,
          (self._img_in, self._img_out), self.log)
    except BaseException:
      for queue in created_queues:
        try:
          queue.cancel_join_thread()
        except Exception as cleanup_error:
          self.log(logging.ERROR, "Could not cancel histogram queue thread "
                                  "join",
                   exc_info=(type(cleanup_error), cleanup_error,
                             cleanup_error.__traceback__))
        try:
          queue.close()
        except Exception as cleanup_error:
          self.log(logging.ERROR, "Could not close histogram queue",
                   exc_info=(type(cleanup_error), cleanup_error,
                             cleanup_error.__traceback__))
      try:
        self.destroy()
      except tk.TclError:
        pass
      raise

    # Attributes containing the several images and histograms
    self._pil_img: Image.Image | None = None
    self._hist: np.ndarray | None = None
    self._pil_hist: Image.Image | None = None
    self._image_tk: ImageTk.PhotoImage | None = None
    self._hist_tk: ImageTk.PhotoImage | None = None

    # Other attributes used in this class
    self._last_upd_t: float | None = None
    self._next_acq_t: float = -float('inf')
    self._n_loops: int = 0

    # Keeping track of the scheduled objects to be able to cancel them later
    self._img_acq_sched_obj: str | None = None
    self._upd_var_sched_obj: str | None = None
    self._shutdown_sched_obj: str | None = None
    self._shutdown_requested: Callable[[], bool] | None = None

    # Settings of the root window
    try:
      self.title(f'Configuration window for the camera: '
                 f'{type(camera).__name__}')
      self.protocol("WM_DELETE_WINDOW", self.finish)
      # Initializing the interface
      self._set_variables()
      self._set_layout()
      self._set_bindings()
      self._add_settings()
      self.update_idletasks()
    except BaseException:
      try:
        self.stop()
      except Exception as cleanup_error:
        self.log(logging.ERROR, "Could not clean up partial configuration",
                 exc_info=(type(cleanup_error), cleanup_error,
                           cleanup_error.__traceback__))
      raise

    # Attribute used only for unit testing, do not use otherwise
    self._testing: bool = False

  def start(self) -> None:
    """Starts the histogram worker and schedules preview updates.

    This method does not wait for the window to close. Use
    :meth:`run() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.run>` for the complete workflow, or manage the toolkit's
    event loop yourself when using
    :meth:`start() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.start>`. The window can be started only once. A closed
    window cannot be restarted.

    Raises:
      RuntimeError: If configuration resources have already been closed.

    .. versionadded:: 1.5.10
    .. versionchanged:: 2.0.7 Renamed from *main()* to *start()*
    """

    if self._lifecycle.closed:
      raise RuntimeError("Cannot start a closed configuration window")

    self.log(logging.DEBUG, "Starting histogram processing and preview "
                            "updates")
    # Starting the histogram calculation process
    self._histogram_process.start()
    self._lifecycle.mark_histogram_started()

    self._n_loops = 0
    self._last_upd_t = time()
    self._next_acq_t = -float('inf')

    # Let Tk's event loop handle the first frame and the first FPS update
    if not self._testing:
      self._img_acq_sched_obj = self.after(0, self._img_acq_sched)
      self._upd_var_sched_obj = self.after(500, self._upd_var_sched)

  def run(self) -> None:
    """Runs configuration until user close, failure, or
    :class:`~crappy.blocks.meta_block.block.Block` shutdown.

    Starts acquisition and waits for the window. Resources are released before
    returning or propagating a failure, including failures during startup. A
    shutdown registered with
    :meth:`watch_shutdown() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.watch_shutdown>` bypasses selection validation.

    Raises:
      RuntimeError: If the window has already been closed.
      KeyboardInterrupt: After cleanup, without an error dialog or error log.
    """

    try:
      # Catch failures at constructor-time
      self._lifecycle.raise_if_failed()
      if self._window_closed:
        raise RuntimeError("Cannot run a closed configuration window")
      self._check_shutdown()
      if not self._window_closed:
        self.start()
        self.wait_window(self)
    except BaseException as error:
      try:
        self.stop()
      except Exception as cleanup_error:
        self.log(logging.ERROR, "Could not clean up configuration window",
                 exc_info=(type(cleanup_error), cleanup_error,
                           cleanup_error.__traceback__))
      if not isinstance(error, KeyboardInterrupt):
        self._lifecycle.raise_if_failed()
      raise

    # Normal termination path
    self.stop()
    self._lifecycle.raise_if_failed()

  def watch_shutdown(self, requested: Callable[[], bool]) -> None:
    """Registers a shutdown condition, regularly checked by the tkinter event
    loop.

    The condition normally performs checks on synchronization objects owned by
    a :class:`~crappy.blocks.meta_block.block.Block`. A true result closes
    without validating or finalizing the current selection.

    Args:
      requested: :obj:`~collections.abc.Callable` returning :obj:`True` when
        the :class:`~crappy.blocks.meta_block.block.Block` is stopping or
        another :class:`~crappy.blocks.meta_block.block.Block` has failed
        during preparation.
    """

    self._shutdown_requested = requested

  def _check_shutdown(self) -> None:
    """Poll the :class:`~crappy.blocks.meta_block.block.Block`'s shutdown
    condition from Tk's event loop."""

    self._shutdown_sched_obj = None

    if self._window_closed or self._shutdown_requested is None:
      return

    if self._shutdown_requested():
      self.log(logging.DEBUG, "Closing configuration after Block shutdown "
                              "request")
      self.stop()
    else:
      self._shutdown_sched_obj = self.after(25, self._check_shutdown)

  def report_callback_exception(self,
                                exc: type[BaseException],
                                val: BaseException,
                                tb: TracebackType | None) -> None:
    """Retain a Tk callback failure, show it, and close the window safely.

    Keyboard interrupts close silently and are re-raised by :meth:`run`.

    .. versionadded:: 2.0.0
    """

    self._lifecycle.record_callback_failure(val, tb)
    try:
      if not isinstance(val, KeyboardInterrupt):
        showerror("Error!", message=f"{exc.__name__}\n{val}")
    except Exception as dialog_error:
      self.log(logging.ERROR, "Could not display configuration error",
               exc_info=(type(dialog_error), dialog_error,
                         dialog_error.__traceback__))
    finally:
      self._lifecycle.request_close(self.stop)

  def finish(self) -> None:
    """Validates and finalizes a user-requested close.

    An invalid selection logs a warning and shows the reason, leaving the
    window open. A valid selection calls
    :meth:`_on_valid_close() <crappy.tool.camera_config.base.camera_config.\
CameraConfig._on_valid_close>` before
    :meth:`stop() <crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig.stop>`. Pending
    :class:`~crappy.camera.meta_camera.camera.Camera` setting edits are not
    applied here. If the :class:`~crappy.blocks.meta_block.block.Block`
    requests shutdown, validation and finalization are skipped.

    .. versionadded:: 2.0.0
    """

    # Finish earlier in case the windows is forcibly closed by parent Block
    if self._shutdown_requested is not None and self._shutdown_requested():
      self.stop()
      return

    # Prevent closing in case something isn't configured yet
    if (reason := self._validate_close()) is not None:
      self.log(logging.WARNING, reason)
      showerror("Error !", message=reason)
      return

    # Otherwise perform closing action and stop
    self._on_valid_close()
    self.log(logging.INFO, "Camera configuration validated")
    self.stop()

  def stop(self) -> None:
    """Closes the window and releases histogram resources without validation.

    Cancels scheduled updates and closes the histogram queues. Repeated calls
    are safe, including before acquisition starts. The
    :class:`~crappy.camera.meta_camera.camera.Camera` object remains open and
    owned by the :class:`~crappy.blocks.meta_block.block.Block`.

    .. versionadded:: 2.0.0
    """

    # If the window is already closed, making sure the resources are released
    if self._window_closed:
      self._lifecycle.close_resources()
      return

    self._window_closed = True
    self.log(logging.DEBUG, "Closing camera configuration and releasing "
                            "histogram resources")
    try:
      if self._img_acq_sched_obj is not None:
        try:
          self.after_cancel(self._img_acq_sched_obj)
        except tk.TclError:
          pass
      if self._upd_var_sched_obj is not None:
        try:
          self.after_cancel(self._upd_var_sched_obj)
        except tk.TclError:
          pass
      if self._shutdown_sched_obj is not None:
        try:
          self.after_cancel(self._shutdown_sched_obj)
        except tk.TclError:
          pass

    finally:
      try:
        self.destroy()
      except tk.TclError as error:
        self.log(logging.ERROR, "Cannot destroy the configuration window",
                 exc_info=(type(error), error, error.__traceback__))
      finally:
        self._lifecycle.close_resources()

  def _img_acq_sched(self) -> None:
    """Acquire a frame when due, then schedule the next acquisition."""

    now = monotonic()
    if self._max_freq is None or now >= self._next_acq_t:
      if self._max_freq is not None:
        self._next_acq_t = now + 1 / self._max_freq
      self._update_img()
      # Camera drivers may have reloaded settings while acquiring the frame
      self._sync_setting_controls()

    if not self._testing and not self._window_closed:
      # Sleep until the next frame is due, or poll as fast as possible
      if self._max_freq is None:
        delay = 1
      else:
        delay = max(1, ceil(1000 * (self._next_acq_t - monotonic())))
      self._img_acq_sched_obj = self.after(delay, self._img_acq_sched)

  def _upd_var_sched(self) -> None:
    """Updates the GUI indicators, and plans the next indicators update."""

    # Planning the next update
    if not self._testing:
      self._upd_var_sched_obj = self.after(500, self._upd_var_sched)

    # Updating the indicators in the GUI
    elapsed = time() - self._last_upd_t
    self._display_state.fps = self._n_loops / elapsed if elapsed > 0 else 0.0
    self._n_loops = 0
    self._last_upd_t = time()
    self._sync_indicator_labels()

  def _set_layout(self) -> None:
    """Creates and places the different elements of the display on the GUI."""

    self.log(logging.DEBUG, "Setting the interface layout")

    # The main frame of the window
    self._main_frame = tk.Frame()
    self._main_frame.pack(fill='both', expand=True)

    # The frame containing the image and the histogram
    self._graphical_frame = tk.Frame(self._main_frame)
    self._graphical_frame.pack(expand=True, fill="both", anchor="w",
                               side="left")

    # The image row will expand 4 times as fast as the histogram row
    self._graphical_frame.columnconfigure(0, weight=1)
    self._graphical_frame.rowconfigure(0, weight=1)
    self._graphical_frame.rowconfigure(1, weight=4)

    # Adapting the default dimension of the GUI according to the screen size
    screen_width = self.winfo_screenwidth()
    screen_height = self.winfo_screenheight()
    if screen_width < 1600 or screen_height < 900:
      min_width, min_height = 600, 450
    else:
      min_width, min_height = 800, 600

    # The label containing the histogram
    self._hist_canvas = tk.Canvas(self._graphical_frame, height=80,
                                  width=min_width, highlightbackground='black',
                                  highlightthickness=1)
    self._hist_canvas.grid(row=0, column=0, sticky='nsew')

    # The label containing the image
    self._img_canvas = tk.Canvas(self._graphical_frame, width=min_width,
                                 height=min_height)
    self._img_canvas.grid(row=1, column=0, sticky='nsew')

    # The frame containing the information on the image and the settings
    self._text_frame = tk.Frame(self._main_frame, highlightbackground='black',
                                highlightthickness=1)
    self._text_frame.pack(expand=True, fill='y', anchor='ne')

    # The frame containing the information on the image
    self._info_frame = tk.Frame(self._text_frame, highlightbackground='black',
                                highlightthickness=1)
    self._info_frame.pack(expand=False, fill='both', anchor='n', side='top',
                          ipady=2)

    # The information on the image
    self._fps_label = tk.Label(self._info_frame, textvariable=self._fps_txt)
    self._fps_label.pack(expand=False, fill='none', anchor='n', side='top')

    self._auto_range_button = tk.Checkbutton(
        self._info_frame, text='Auto range', variable=self._auto_range_var,
        command=self._on_auto_range_toggle)
    self._auto_range_button.pack(expand=False, fill='none', anchor='n',
                                 side='top')

    self._auto_apply_button = tk.Checkbutton(
        self._info_frame, text='Auto apply', variable=self._auto_apply_var,
        command=self._on_auto_apply_toggle)
    self._auto_apply_button.pack(expand=False, fill='none', anchor='n',
                                 side='top')

    self._min_max_label = tk.Label(self._info_frame,
                                   textvariable=self._min_max_pix_txt)
    self._min_max_label.pack(expand=False, fill='none', anchor='n', side='top')

    self._bits_label = tk.Label(self._info_frame, textvariable=self._bits_txt)
    self._bits_label.pack(expand=False, fill='none', anchor='n', side='top')

    self._zoom_label = tk.Label(self._info_frame, textvariable=self._zoom_txt)
    self._zoom_label.pack(expand=False, fill='none', anchor='n', side='top')

    self._reticle_label = tk.Label(self._info_frame,
                                   textvariable=self._reticle_txt)
    self._reticle_label.pack(expand=False, fill='none', anchor='n', side='top')

    # The frame containing the settings, the message and the update button
    self._sets_frame = tk.Frame(self._text_frame)
    self._sets_frame.pack(expand=True, fill='both', anchor='e', side='top')

    # Tha label warning the user
    self._validate_text = tk.Label(
      self._sets_frame,
      text='To validate the choice of the settings and start the test, simply '
           'close this window.',
      fg='#f00', wraplength=300)
    self._validate_text.pack(expand=False, fill='none', ipadx=5, ipady=5,
                             padx=5, pady=5, anchor='n', side='top')

    # The update button
    self._create_buttons()

    # The frame containing the settings
    self._settings_frame = tk.Frame(self._sets_frame,
                                    highlightbackground='black',
                                    highlightthickness=1)
    self._settings_frame.pack(expand=True, fill='both', anchor='n', side='top')

    # The canvas containing the settings
    self._settings_canvas = tk.Canvas(self._settings_frame)
    self._settings_canvas.pack(expand=True, fill='both', anchor='w',
                               side='left')
    self._canvas_frame = tk.Frame(self._settings_canvas)
    self._id = self._settings_canvas.create_window(
      0, 0, window=self._canvas_frame, anchor='nw',
      width=self._settings_canvas.winfo_reqwidth(), tags='canvas window')

    # Creating the scrollbar
    self._vbar = tk.Scrollbar(self._settings_frame, orient="vertical")
    self._vbar.pack(expand=True, fill='y', side='right')
    self._vbar.config(command=self._custom_yview)

    # Associating the scrollbar with the settings canvas
    self._settings_canvas.config(yscrollcommand=self._vbar.set)

  def _create_buttons(self) -> None:
    """Create the Apply button and any backend-independent extra actions."""

    self._apply_button = tk.Button(self._sets_frame, text="Apply Settings",
                                   command=self._update_settings)
    self._update_button = self._apply_button  # For historical compatibility
    self._apply_button.pack(expand=False, fill='none', ipadx=5, ipady=5,
                            padx=5, pady=5, anchor='n', side='top')

    # Add any extra button requested by children classes
    self._action_buttons: dict[str, tk.Button] = dict()
    for action in self._extra_actions():
      if action.id in self._action_buttons:
        raise ValueError(f"Duplicate configuration action: {action.id}")
      button = tk.Button(self._sets_frame, text=action.label,
                         command=action.callback)
      button.pack(expand=False, fill='none', ipadx=5, ipady=5,
                  padx=5, pady=5, anchor='n', side='top')
      self._action_buttons[action.id] = button

  def _custom_yview(self, *args) -> None:
    """Custom handling of the settings canvas scrollbar, that does nothing
    if the entire canvas is already visible."""

    if self._settings_canvas.yview() == (0., 1.):
      return
    self._settings_canvas.yview(*args)

  def _set_bindings(self) -> None:
    """Sets the bindings for the different events triggered by the user."""

    self.log(logging.DEBUG, "Setting the interface bindings")

    # Bindings for the settings canvas
    self._settings_canvas.bind("<Configure>", self._configure_canvas)
    self._settings_frame.bind('<Enter>', self._bind_mouse)
    self._settings_frame.bind('<Leave>', self._unbind_mouse)

    # Different mousewheel handling depending on the platform
    if system() == "Linux":
      self._img_canvas.bind('<4>', self._on_wheel_img)
      self._img_canvas.bind('<5>', self._on_wheel_img)
    else:
      self._img_canvas.bind('<MouseWheel>', self._on_wheel_img)

    # Bindings for the image canvas
    self._img_canvas.bind('<Motion>', self._update_coord)
    self._img_canvas.bind('<ButtonPress-3>', self._start_move)
    self._img_canvas.bind('<B3-Motion>', self._move)

    # Each canvas reports its own final size
    self._img_canvas.bind("<Configure>", self._on_img_resize)
    self._hist_canvas.bind("<Configure>", self._on_hist_resize)

  def _bind_mouse(self, _: tk.Event) -> None:
    """Binds the mousewheel to the settings canvas scrollbar when the user
    hovers over the canvas."""

    self.log(logging.DEBUG, "Binding the mouse wheel to the settings canvas")

    if system() == "Linux":
      self._settings_frame.bind_all('<4>', self._on_wheel_settings)
      self._settings_frame.bind_all('<5>', self._on_wheel_settings)
    else:
      self._settings_frame.bind_all('<MouseWheel>', self._on_wheel_settings)

  def _unbind_mouse(self, _: tk.Event) -> None:
    """Unbinds the mousewheel to the settings canvas scrollbar when the mouse
    leaves the canvas."""

    self.log(logging.DEBUG, "Unbinding the mouse wheel from the settings "
                            "canvas")

    self._settings_frame.unbind_all('<4>')
    self._settings_frame.unbind_all('<5>')
    self._settings_frame.unbind_all('<MouseWheel>')

  def _configure_canvas(self, event: tk.Event) -> None:
    """Adjusts the size of the scrollbar according to the size of the settings
    canvas whenever it is being resized."""

    self.log(logging.DEBUG, "The settings canvas has been resized")

    # Adjusting the height of the settings window inside the canvas
    self._settings_canvas.itemconfig(
      self._id, width=event.width,
      height=self._canvas_frame.winfo_reqheight())

    # Setting the scroll region according to the height of the settings window
    self._settings_canvas.configure(
      scrollregion=(0, 0, self._canvas_frame.winfo_reqwidth(),
                    self._canvas_frame.winfo_reqheight()))

  def _on_wheel_settings(self, event: tk.Event) -> None:
    """Scrolls the canvas up or down upon wheel motion."""

    # Do nothing if the entire canvas is already visible
    if self._settings_canvas.yview() == (0., 1.):
      return

    # Different wheel management in Windows and Linux
    if system() == "Linux":
      delta = 1 if event.num == 4 else -1
    else:
      delta = (event.delta > 0) - (event.delta < 0)

    if not delta:
      return

    self._settings_canvas.yview_scroll(-delta, "units")

  def _on_wheel_img(self, event: tk.Event) -> None:
    """Translate a Tk wheel event into an image-coordinate zoom request."""

    if system() == "Linux":
      direction = 1 if event.num == 4 else -1
    else:
      direction = (event.delta > 0) - (event.delta < 0)

    if self._zoom_at(event.x, event.y, direction):
      self._on_img_resize()
      self._sync_indicator_labels()

  def _update_coord(self, event: tk.Event) -> None:
    """Translate Tk motion into a display-coordinate reticle update."""

    if self._point_at(event.x, event.y):
      self._sync_indicator_labels()

  def _start_move(self, event: tk.Event) -> None:
    """Translate a Tk right-button press into a pan start."""

    self._begin_pan(event.x, event.y)

  def _move(self, event: tk.Event) -> None:
    """Translate a Tk right-button drag into a pan update."""

    self._pan_to(event.x, event.y)

  def _check_event_pos(self, event: tk.Event) -> bool:
    """Tk compatibility adapter for subclass event handlers."""

    return self._is_on_image(event.x, event.y)

  def _add_settings(self) -> None:
    """Adds the settings of the camera to the GUI."""

    self.log(logging.DEBUG, "Adding the camera settings to the interface")

    for setting in self._setting_manager.local_settings:
      self._add_setting_control(setting)

    # First, sort the settings by type for a nicer display
    sort_sets = sorted(self._camera.settings.values(),
                       key=lambda setting_: setting_.type.__name__)

    for cam_set in sort_sets:
      self._add_setting_control(cam_set)

  def _add_setting_control(self, setting: CameraSetting) -> None:
    """Create a Tk control for a supported camera or local setting."""

    if isinstance(setting, CameraBoolSetting):
      self._add_bool_setting(setting)
    elif isinstance(setting, CameraScaleSetting):
      self._add_slider_setting(setting)
    elif isinstance(setting, CameraChoiceSetting):
      self._add_choice_setting(setting)

  def _register_setting_control(self,
                                setting: CameraSetting,
                                control: _TkSettingControl) -> None:
    """Own a Tk control, retaining support for manually added local
    settings."""

    self._setting_manager.register_local(setting)
    self._setting_controls[setting] = control

  def _add_bool_setting(self, cam_set: CameraBoolSetting) -> None:
    """Adds a setting represented by a checkbutton."""

    self.log(logging.DEBUG, f"Adding the boolean setting {cam_set.name}")

    variable = tk.BooleanVar(value=cam_set.value)
    widget = tk.Checkbutton(self._canvas_frame,
                            text=cam_set.name,
                            variable=variable,
                            command=self._auto_apply_settings)

    widget.pack(anchor='w', side='top', expand=False, fill='none',
                padx=5, pady=2)
    self._register_setting_control(cam_set, _TkSettingControl(
      variable, widget, cam_set.revision))

  def _add_slider_setting(self, cam_set: CameraScaleSetting) -> None:
    """Adds a setting represented by a scale bar."""

    self.log(logging.DEBUG, f"Adding the slider setting {cam_set.name}")

    # The scale bar is slightly different if the setting type is int or float
    if cam_set.type == int:
      variable = tk.IntVar(value=int(cam_set.value))
    else:
      variable = tk.DoubleVar(value=cam_set.value)

    widget = tk.Scale(self._canvas_frame,
                      label=f'{cam_set.name} :',
                      variable=variable,
                      resolution=cam_set.step,
                      orient='horizontal',
                      from_=cam_set.lowest,
                      to=cam_set.highest)

    widget.bind("<ButtonRelease-1>", self._auto_apply_settings)

    widget.pack(anchor='center', side='top', expand=False,
                fill='x', padx=5, pady=2)
    self._register_setting_control(cam_set, _TkSettingControl(
      variable, widget, cam_set.revision))

  def _add_choice_setting(self, cam_set: CameraChoiceSetting) -> None:
    """Adds a setting represented by a :obj:`list` of radio buttons."""

    self.log(logging.DEBUG, f"Adding the choice setting {cam_set.name}")

    variable = tk.StringVar(value=cam_set.value)
    frame = tk.Frame(self._canvas_frame)
    frame.pack(anchor='w', side='top', expand=False, fill='x')
    label = tk.Label(frame, text=f'{cam_set.name} :')
    label.pack(anchor='w', side='top', expand=False, fill='none',
               padx=12, pady=2)

    buttons = []
    for value in cam_set.choices:
      widget = tk.Radiobutton(frame,
                              text=value,
                              variable=variable,
                              value=value,
                              command=self._auto_apply_settings)

      widget.pack(anchor='w', side='top', expand=False,
                  fill='none', padx=5, pady=2)
      buttons.append(widget)

    self._register_setting_control(cam_set, _TkSettingControl(
      variable, buttons, cam_set.revision, frame))

  def _sync_setting_controls(self,
                             settings: Iterable[CameraSetting] | None = None
                             ) -> None:
    """Copy changed setting values and metadata into their Tk controls.

    This is model-to-view synchronization, not the path that applies edits.
    It runs after each setting write and image acquisition so a setter or
    camera driver can reload another setting. An unchanged revision
    leaves that control alone, preserving a pending user edit before Apply.
    If ``settings`` is given, only those settings are checked.
    """

    for setting in (self._setting_controls if settings is None else settings):
      control = self._setting_controls.get(setting)
      if control is None:
        continue
      if control.revision == setting.revision:
        continue

      self.log(logging.DEBUG, f"Synchronizing control for setting "
                              f"{setting.name}")
      if isinstance(setting, CameraScaleSetting):
        widget = control.widget
        if not isinstance(widget, tk.Scale):
          raise RuntimeError("Scale setting has no scale control")
        if setting.type is int and not isinstance(control.variable, tk.IntVar):
          control.variable = tk.IntVar()
          widget.configure(variable=control.variable)
        elif (setting.type is float and
              not isinstance(control.variable, tk.DoubleVar)):
          control.variable = tk.DoubleVar()
          widget.configure(variable=control.variable)
        widget.configure(from_=setting.lowest, to=setting.highest,
                         resolution=setting.step)

      elif isinstance(setting, CameraChoiceSetting):
        buttons = control.widget
        if not isinstance(buttons, list) or control.frame is None:
          raise RuntimeError("Choice setting has no radio controls")
        for i, choice in enumerate(setting.choices):
          if i >= len(buttons):
            button = tk.Radiobutton(control.frame, variable=control.variable,
                                    command=self._auto_apply_settings)
            button.pack(anchor='w', side='top', expand=False,
                        fill='none', padx=5, pady=2)
            buttons.append(button)
          buttons[i].configure(value=choice, text=choice, state='normal')
        for button in buttons[len(setting.choices):]:
          button.configure(value='', text='', state='disabled')

      control.variable.set(setting.value)
      control.revision = setting.revision

  def _set_variables(self) -> None:
    """Create Tk-only variables for rendering the plain display state."""

    self.log(logging.DEBUG, "Setting the interface variables")

    self._auto_range_var = tk.BooleanVar(value=self._display_state.auto_range)
    self._auto_apply_var = tk.BooleanVar(value=self._display_state.auto_apply)
    self._fps_txt = tk.StringVar()
    self._min_max_pix_txt = tk.StringVar()
    self._bits_txt = tk.StringVar()
    self._zoom_txt = tk.StringVar()
    self._reticle_txt = tk.StringVar()
    self._sync_indicator_labels()

  def _sync_indicator_labels(self) -> None:
    """Render the current ordinary state in Tk label variables."""

    state = self._display_state
    self._fps_txt.set(f'fps = {state.fps:.2f}\n'
                      f'(might be lower in this GUI than actual)')
    self._min_max_pix_txt.set(f'min: {state.min_pixel:d}, '
                              f'max: {state.max_pixel:d}')
    self._bits_txt.set(f'Detected bits: {state.detected_bits:d}')
    self._zoom_txt.set(f'Zoom: {state.zoom_percent:.1f}%')
    self._reticle_txt.set(f'X: {state.reticle_x:d}, '
                          f'Y: {state.reticle_y:d}, '
                          f'V: {state.reticle_value:d}')

  def _on_auto_range_toggle(self) -> None:
    """Translate the Tk checkbutton state into an ordinary option value."""

    self._display_state.auto_range = bool(self._auto_range_var.get())
    self.log(logging.DEBUG, "Auto range " +
             ("enabled" if self._display_state.auto_range else "disabled"))

  def _on_auto_apply_toggle(self) -> None:
    """Translate the Tk checkbutton state and update the Apply button."""

    self._display_state.auto_apply = bool(self._auto_apply_var.get())
    self.log(logging.DEBUG, "Auto apply " +
             ("enabled" if self._display_state.auto_apply else "disabled"))
    self._apply_button['state'] = ('disabled' if self._display_state.auto_apply
                                   else 'normal')

  def _update_settings(self) -> None:
    """Apply local settings, then camera settings, through the shared manager.

    The Apply Settings button and auto-apply both enter here. The Tk backend
    supplies one requested value at a time so a dependent reload can refresh
    a later control before its value is read.
    """

    self.log(logging.DEBUG, "Applying camera configuration settings")
    for setting in self._setting_manager.settings:
      self._apply_setting(setting)

  def _apply_setting(self, setting: CameraSetting) -> None:
    """Read one Tk control, delegate its write, then refresh changed
    controls."""

    self._sync_setting_controls()
    control = self._setting_controls.get(setting)
    if control is None:
      return

    result = self._setting_manager.apply(setting, control.variable.get())
    self._sync_setting_controls(result.changed)

  def _auto_apply_settings(self, *_: tk.Event):
    """Applies pending edits when Auto apply is enabled.

    Checkboxes and radio buttons apply on activation. Scales apply when the
    slider is released. All settings use the shared application order.
    """

    if self._display_state.auto_apply:
      self._update_settings()

  def _read_image_geometry(self) -> None:
    """Translate the Tk image-canvas size into ordinary display geometry."""

    self._display_geometry.width = self._img_canvas.winfo_width()
    self._display_geometry.height = self._img_canvas.winfo_height()

  def _read_histogram_geometry(self) -> None:
    """Copy the Tk histogram-canvas size before resizing its image."""

    self._hist_width = self._hist_canvas.winfo_width()
    self._hist_height = self._hist_canvas.winfo_height()

  def _resize_img(self) -> None:
    """Resizes the received image so that it fits in the display area and
    complies with the chosen zoom level."""

    if self._img is None:
      return

    self.log(logging.DEBUG, "Resizing the image to fit in the window")

    # First, apply the current zoom level
    # The width and height values are inverted in NumPy
    img_height, img_width, *_ = self._img.shape
    y_min_pix = int(img_height * self._zoom_values.y_low)
    y_max_pix = int(img_height * self._zoom_values.y_high)
    x_min_pix = int(img_width * self._zoom_values.x_low)
    x_max_pix = int(img_width * self._zoom_values.x_high)
    zoomed_img = self._img[y_min_pix: y_max_pix, x_min_pix: x_max_pix]

    if not zoomed_img.size:
      self._pil_img = None
      self._display_geometry.image_width = 0
      self._display_geometry.image_height = 0
      return

    # Creating the pillow image from the zoomed numpy array
    pil_img = Image.fromarray(zoomed_img)

    new_width, new_height = self._display_geometry.fit(pil_img.width,
                                                       pil_img.height)
    if not new_width or not new_height:
      self._pil_img = None
      self._display_geometry.image_width = 0
      self._display_geometry.image_height = 0
      return

    self._pil_img = pil_img.resize((new_width, new_height))
    self._display_geometry.image_width = new_width
    self._display_geometry.image_height = new_height

  def _display_img(self) -> None:
    """Displays the image in the center of the image canvas."""

    if self._pil_img is None:
      return

    self.log(logging.DEBUG, "Displaying the image")

    self._image_tk = ImageTk.PhotoImage(self._pil_img)
    self._img_canvas.create_image(int(self._display_geometry.width / 2),
                                  int(self._display_geometry.height / 2),
                                  anchor='center', image=self._image_tk)

  def _on_img_resize(self, _: tk.Event | None = None) -> None:
    """Resizes the image and updates the display when the zoom level has
    changed or the GUI has been resized."""

    self.log(logging.DEBUG, "The image canvas was resized")

    self._read_image_geometry()
    self._draw_overlay()

    self._resize_img()
    self._display_img()

  def _calc_hist(self) -> None:
    """Calculates the histogram of the current image."""

    if self._original_img is None:
      return

    # Don't calculate histogram if a calculation is already running
    if self._processing_event.is_set():
      self.log(logging.DEBUG, "A calculation is running for the histogram, "
                              "not sending image for calculation")

    # If no calculation is running, sending a new image for calculation
    else:
      # Reshaping the image before sending to the histogram process
      self.log(logging.DEBUG, "Preparing image for histogram calculation")
      hist_img = Image.fromarray(self._original_img)
      if hist_img.width > 320 or hist_img.height > 240:
        factor = min(320 / hist_img.width, 240 / hist_img.height)
        hist_img = hist_img.resize((max(int(hist_img.width * factor), 1),
                                    max(int(hist_img.height * factor), 1)))
      # The histogram is calculated on a grey level image
      if len(self._original_img.shape) == 3:
        hist_img = hist_img.convert('L')

      # Sending the image to the histogram process
      self.log(logging.DEBUG, "Sending image for histogram calculation")
      self._img_in.put_nowait((hist_img, self._display_state.auto_range,
                               self._low_thresh, self._high_thresh))

    # Always check for completed output, including while the process is already
    # calculating the next histogram
    try:
      while True:
        self._hist = self._img_out.get_nowait()
        self.log(logging.DEBUG, "Received histogram from histogram process")
    except Empty:
      pass

  def _resize_hist(self) -> None:
    """Resizes the histogram image to make it fit in the GUI."""

    if self._hist is None:
      return

    self.log(logging.DEBUG, "Resizing the histogram to fit in the window")

    pil_hist = Image.fromarray(self._hist)
    if self._hist_width <= 0 or self._hist_height <= 0:
      self._pil_hist = None
      return

    self._pil_hist = pil_hist.resize((self._hist_width, self._hist_height))

  def _display_hist(self) -> None:
    """Displays the histogram image in the GUI."""

    if self._pil_hist is None:
      return

    self.log(logging.DEBUG, "Displaying the histogram")

    self._hist_tk = ImageTk.PhotoImage(self._pil_hist)
    self._hist_canvas.create_image(int(self._hist_width / 2),
                                   int(self._hist_height / 2),
                                   anchor='center', image=self._hist_tk)

  def _on_hist_resize(self, _: tk.Event | None = None) -> None:
    """Resizes the histogram and updates the display when the GUI has been
    resized."""

    self._read_histogram_geometry()
    self._resize_hist()
    self._display_hist()

  def _update_img(self) -> None:
    """Acquire an image through the core and render it in Tk."""

    self.log(logging.DEBUG, "Updating the image")

    if not self._acquire_image():
      return

    self._read_image_geometry()
    self._read_histogram_geometry()
    self._draw_overlay()
    self._resize_img()

    self._calc_hist()
    self._resize_hist()

    self._display_img()
    self._display_hist()

    self._update_pixel_value()
    self._sync_indicator_labels()

  def _draw_overlay(self) -> None:
    """Method meant to be used by subclasses for drawing an overlay on top of
    the image to display."""

    ...

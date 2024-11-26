# coding: utf-8

import tkinter as tk
from tkinter.messagebox import showerror
from platform import system
import numpy as np
from time import time, sleep
from typing import Optional, Union
from pkg_resources import resource_string
from io import BytesIO
import logging
from multiprocessing import current_process, Event, Queue
from multiprocessing.queues import Queue as MPQueue

from .config_tools import Zoom, HistogramProcess
from ...camera.meta_camera.camera_setting import CameraBoolSetting, \
  CameraChoiceSetting, CameraScaleSetting
from ...camera.meta_camera import Camera
from ..._global import OptionalModule

try:
  from PIL import ImageTk, Image
except (ModuleNotFoundError, ImportError):
  ImageTk = OptionalModule("pillow")
  Image = OptionalModule("pillow")


class CameraConfig(tk.Tk):
  """This class is a GUI allowing the user to visualize the images from a
  :class:`~crappy.camera.Camera` before a Crappy test starts, and to tune the
  settings of the Camera.

  It is meant to be user-friendly and interactive. It is possible to zoom on
  the image using the mousewheel, and to move on the zoomed image by
  right-clicking and dragging.

  In addition to the image, the interface also displays a histogram of the
  pixel values, an FPS counter, a detected bits counter, the minimum and
  maximum pixel values, and the value and position of the pixel currently under
  the mouse. A checkbox allows auto-adjusting the pixel range to get a better
  contrast.

  This class is used as is by the :class:`~crappy.blocks.Camera`, but also 
  subclassed to provide more specific functionalities to other camera-related 
  Blocks like :class:`~crappy.blocks.VideoExtenso` or 
  :class:`~crappy.blocks.DICVE`.
  
  This class is a child of :obj:`tkinter.Tk`. It relies on the 
  :class:`~crappy.tool.camera_config.config_tools.Zoom` and 
  :class:`~crappy.tool.camera_config.config_tools.HistogramProcess` tools. It
  also interacts with instances of the 
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` class.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Camera_config* to *CameraConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: Optional[int],
               max_freq: Optional[float]) -> None:
    """Initializes the interface and displays it.

    Args:
      camera: The :class:`~crappy.camera.Camera` object in charge of acquiring 
        the images.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      max_freq: The maximum frequency this window is allowed to loop at. It is
        simply the ``freq`` attribute of the :class:`~crappy.blocks.Camera`
        Block.

        .. versionadded:: 2.0.0
    """

    super().__init__()
    self._camera = camera
    self.shape: Optional[Union[tuple[int, int], tuple[int, int, int]]] = None
    self.dtype = None
    self._logger: Optional[logging.Logger] = None

    # Instantiating objects for the process managing the histogram calculation
    self._stop_event = Event()
    self._processing_event = Event()
    self._img_in = Queue(maxsize=0)
    self._img_out = Queue(maxsize=0)
    self._histogram_process = HistogramProcess(
        stop_event=self._stop_event, processing_event=self._processing_event,
        img_in=self._img_in, img_out=self._img_out, log_level=log_level,
        log_queue=log_queue)

    # Attributes containing the several images and histograms
    self._img = None
    self._pil_img = None
    self._original_img = None
    self._hist = None
    self._pil_hist = None

    # Other attributes used in this class
    self._low_thresh = None
    self._high_thresh = None
    self._move_x = None
    self._move_y = None
    self._run = True
    self._n_loops = 0
    self._max_freq = max_freq
    self._got_first_img: bool = False

    # Settings for adjusting the behavior of the zoom
    self._zoom_ratio = 0.9
    self._zoom_step = 0
    self._max_zoom_step = 15

    # Settings of the root window
    self.title(f'Configuration window for the camera: {type(camera).__name__}')
    self.protocol("WM_DELETE_WINDOW", self.finish)
    self._zoom_values = Zoom()

    # Initializing the interface
    self._set_variables()
    self._set_traces()
    self._set_layout()
    self._set_bindings()
    self._add_settings()
    self.update()

  def main(self) -> None:
    """Constantly updates the image and the information on the GUI, until asked
    to stop.
    
    .. versionadded:: 1.5.10
    """

    # Starting the histogram calculation process
    self._histogram_process.start()

    self._n_loops = 0
    start_time = time()

    while self._run:
      # Remaining below the max allowed frequency
      if self._max_freq is None or (self._n_loops <
                                    self._max_freq * (time() - start_time)):
        # Update the image, the histogram and the information
        self._update_img()

      # Update the FPS counter
      if time() - start_time > 0.5:
        self._fps_var.set(self._n_loops / (time() - start_time))
        self._n_loops = 0
        start_time = time()

  def log(self, level: int, msg: str) -> None:
    """Record log messages for the CameraConfig window.

    Also instantiates the :obj:`~logging.Logger` when logging the first
    message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def report_callback_exception(self, exc: Exception, val: str, tb) -> None:
    """Method displaying an error message in case an exception is raised in a
    :mod:`tkinter` callback.

    .. versionadded:: 2.0.0
    """

    self._logger.exception(f"Caught exception in {type(self).__name__}: "
                           f"{exc.__name__}({val})", exc_info=tb)
    showerror("Error !", message=f"{exc.__name__}\n{val}")

  def finish(self) -> None:
    """Method called when the user tries to close the configuration window.

    Mostly intended for being overwritten.
    
    .. versionadded:: 2.0.0
    """

    self.stop()

  def stop(self) -> None:
    """Method called for gracefully stopping the GUI.

    Stops the process calculating the histogram, and destroys the GUI.

    .. versionadded:: 2.0.0
    """

    # Stopping the event loop and the histogram process
    self._run = False
    self._stop_event.set()
    sleep(0.1)

    # Killing the histogram process if it's still alive
    if self._histogram_process.is_alive():
      self.log(logging.WARNING, "The histogram process failed to stop, "
                                "killing it !")
      self._histogram_process.terminate()

    self.log(logging.DEBUG, "Destroying the configuration window")

    try:
      self.destroy()
    except tk.TclError:
      self.log(logging.WARNING, "Cannot destroy the configuration window, "
                                "ignoring")

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

    self._auto_range_button = tk.Checkbutton(self._info_frame,
                                             text='Auto range',
                                             variable=self._auto_range)
    self._auto_range_button.pack(expand=False, fill='none', anchor='n',
                                 side='top')

    self._auto_apply_button = tk.Checkbutton(self._info_frame,
                                             text='Auto apply',
                                             variable=self._auto_apply)
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
    """This method is meant to simplify the addition of extra buttons in
    subclasses."""

    self._update_button = tk.Button(self._sets_frame, text="Apply Settings",
                                    command=self._update_settings)
    self._update_button.pack(expand=False, fill='none', ipadx=5, ipady=5,
                             padx=5, pady=5, anchor='n', side='top')

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

    # It's more efficient to bind the resizing to the graphical frame
    self._graphical_frame.bind("<Configure>", self._on_img_resize)
    self._graphical_frame.bind("<Configure>", self._on_hist_resize)

  def _bind_mouse(self, _: tk.Event) -> None:
    """Binds the mousewheel to the settings canvas scrollbar when the user
    hovers over the canvas."""

    self.log(logging.DEBUG, "Binding the mouse to the image canvas")

    if system() == "Linux":
      self._settings_frame.bind_all('<4>', self._on_wheel_settings)
      self._settings_frame.bind_all('<5>', self._on_wheel_settings)
    else:
      self._settings_frame.bind_all('<MouseWheel>', self._on_wheel_settings)

  def _unbind_mouse(self, _: tk.Event) -> None:
    """Unbinds the mousewheel to the settings canvas scrollbar when the mouse
    leaves the canvas."""

    self.log(logging.DEBUG, "Unbinding the mouse from the image canvas")

    self._settings_frame.unbind_all('<4>')
    self._settings_frame.unbind_all('<5>')
    self._settings_frame.unbind_all('<MouseWheel>')

  def _configure_canvas(self, event: tk.Event) -> None:
    """Adjusts the size of the scrollbar according to the size of the settings
    canvas whenever it is being resized."""

    self.log(logging.DEBUG, "The image canvas has been resized")

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
      delta = int(event.delta / abs(event.delta))

    self._settings_canvas.yview_scroll(-delta, "units")

  def _on_wheel_img(self, event: tk.Event) -> None:
    """Zooms in or out on the image upon mousewheel motion.

    Handles the specific cases when the mouse is not on the image, or the
    maximum or minimum zoom levels are reached.
    """

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self.log(logging.DEBUG, "Zooming on the canvas")

    pil_width = self._pil_img.width
    pil_height = self._pil_img.height
    zoom_x_low, zoom_x_high = self._zoom_values.x_low, self._zoom_values.x_high
    zoom_y_low, zoom_y_high = self._zoom_values.y_low, self._zoom_values.y_high

    # Different wheel management in Windows and Linux
    if system() == "Linux":
      delta = 1 if event.num == 4 else -1
    else:
      delta = int(event.delta / abs(event.delta))

    # Handling the cases when the minimum or maximum zoom levels are reached
    self._zoom_step += delta
    if self._zoom_step < 0:
      self._zoom_step = 0
      self._zoom_level.set(100)
      return
    elif self._zoom_step == 0:
      self._zoom_values.reset()
      self._zoom_level.set(100)
      self._on_img_resize()
      return
    elif self._zoom_step > self._max_zoom_step:
      self._zoom_step = self._max_zoom_step
      self._zoom_level.set(100 * (1 / self._zoom_ratio) ** self._max_zoom_step)
      return

    # Correcting the event position to make it relative to the image and not
    # the canvas
    zero_x = (self._img_canvas.winfo_width() - pil_width) / 2
    zero_y = (self._img_canvas.winfo_height() - pil_height) / 2
    corr_x = event.x - zero_x
    corr_y = event.y - zero_y

    # The position of the mouse on the image as a ratio between 0 and 1
    x_ratio = corr_x * (zoom_x_high - zoom_x_low) / pil_width
    y_ratio = corr_y * (zoom_y_high - zoom_y_low) / pil_height

    # Updating the upper and lower limits of the image on the display
    ratio = self._zoom_ratio if delta < 0 else 1 / self._zoom_ratio
    self._zoom_values.update_zoom(x_ratio, y_ratio, ratio)

    # Redrawing the image and updating the information
    self._on_img_resize()
    self._zoom_level.set(100 * (1 / self._zoom_ratio) ** self._zoom_step)

  def _update_coord(self, event: tk.Event) -> None:
    """Updates the coordinates of the pixel pointed by the mouse on the
    image."""

    self.log(logging.DEBUG, "Updating the coordinates of the current pixel")

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    x_coord, y_coord = self._coord_to_pix(event.x, event.y)

    self._x_pos.set(x_coord)
    self._y_pos.set(y_coord)

    self._update_pixel_value()

  def _update_pixel_value(self) -> None:
    """Updates the display of the gray level value of the pixel currently being
    pointed by the mouse."""

    self.log(logging.DEBUG, "Updating the value of the current pixel")

    try:
      self._reticle_val.set(int(np.average(
          self._original_img[self._y_pos.get(), self._x_pos.get()])))
    except IndexError:
      self._x_pos.set(0)
      self._y_pos.set(0)
      self._reticle_val.set(int(np.average(
          self._original_img[self._y_pos.get(), self._x_pos.get()])))

  def _coord_to_pix(self, x: int, y: int) -> tuple[int, int]:
    """Converts the coordinates of the mouse in the GUI referential to
    coordinates on the original image."""

    pil_width = self._pil_img.width
    pil_height = self._pil_img.height
    zoom_x_low, zoom_x_high = self._zoom_values.x_low, self._zoom_values.x_high
    zoom_y_low, zoom_y_high = self._zoom_values.y_low, self._zoom_values.y_high
    img_height, img_width, *_ = self._img.shape

    # Correcting the event position to make it relative to the image and not
    # the canvas
    zero_x = (self._img_canvas.winfo_width() - pil_width) / 2
    zero_y = (self._img_canvas.winfo_height() - pil_height) / 2
    corr_x = x - zero_x
    corr_y = y - zero_y

    # Convert the relative coordinate of the mouse on the display to coordinate
    # of the mouse on the original image
    x_disp = corr_x / pil_width * (zoom_x_high - zoom_x_low) * img_width
    y_disp = corr_y / pil_height * (zoom_y_high - zoom_y_low) * img_height

    # The coordinate of the upper left corner of the displayed image
    # (potentially zoomed) on the original image
    x_trim = zoom_x_low * img_width
    y_trim = zoom_y_low * img_height

    return min(int(x_disp + x_trim),
               img_width - 1), min(int(y_disp + y_trim), img_height - 1)

  def _start_move(self, event: tk.Event) -> None:
    """Stores the position of the mouse upon left-clicking on the image."""

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self.log(logging.DEBUG, "Drag started")

    # Stores the position of the mouse relative to the top left corner of the
    # image
    zero_x = (self._img_canvas.winfo_width() - self._pil_img.width) / 2
    zero_y = (self._img_canvas.winfo_height() - self._pil_img.height) / 2
    self._move_x = event.x - zero_x
    self._move_y = event.y - zero_y

  def _move(self, event: tk.Event) -> None:
    """Drags the image upon prolonged left-clik and drag from the user."""

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self.log(logging.DEBUG, "Drag ended")

    pil_width = self._pil_img.width
    pil_height = self._pil_img.height
    zoom_x_low, zoom_x_high = self._zoom_values.x_low, self._zoom_values.x_high
    zoom_y_low, zoom_y_high = self._zoom_values.y_low, self._zoom_values.y_high

    # Getting the position delta, in the coordinates of the display
    zero_x = (self._img_canvas.winfo_width() - pil_width) / 2
    zero_y = (self._img_canvas.winfo_height() - pil_height) / 2
    delta_x_disp = self._move_x - (event.x - zero_x)
    delta_y_disp = self._move_y - (event.y - zero_y)

    # Converting the position delta to a ratio between 0 and 1 relative to the
    # size of the original image
    delta_x = delta_x_disp * (zoom_x_high - zoom_x_low) / pil_width
    delta_y = delta_y_disp * (zoom_y_high - zoom_y_low) / pil_height

    # Actually updating the display
    self._zoom_values.update_move(delta_x, delta_y)

    # Resetting the original position, otherwise the drag never ends
    self._move_x = event.x - zero_x
    self._move_y = event.y - zero_y

  def _check_event_pos(self, event: tk.Event) -> bool:
    """Checks whether the mouse is on the image, and not between the image and
    the border of the canvas. Returns :obj:`True` if it is on the image,
    :obj:`False` otherwise."""

    if self._pil_img is None:
      return False

    if abs(event.x -
           self._img_canvas.winfo_width() / 2) > self._pil_img.width / 2:
      return False
    if abs(event.y -
           self._img_canvas.winfo_height() / 2) > self._pil_img.height / 2:
      return False

    return True

  def _add_settings(self) -> None:
    """Adds the settings of the camera to the GUI."""

    self.log(logging.DEBUG, "Adding the camera settings to the interface")

    # First, sort the settings by type for a nicer display
    sort_sets = sorted(self._camera.settings.values(),
                       key=lambda setting: setting.type.__name__)

    for cam_set in sort_sets:
      if isinstance(cam_set, CameraBoolSetting):
        self._add_bool_setting(cam_set)
      elif isinstance(cam_set, CameraScaleSetting):
        self._add_slider_setting(cam_set)
      elif isinstance(cam_set, CameraChoiceSetting):
        self._add_choice_setting(cam_set)

  def _add_bool_setting(self, cam_set: CameraBoolSetting) -> None:
    """Adds a setting represented by a checkbutton."""

    self.log(logging.DEBUG, f"Adding the boolean setting {cam_set.name}")

    cam_set.tk_var = tk.BooleanVar(value=cam_set.value)
    cam_set.tk_obj = tk.Checkbutton(self._canvas_frame,
                                    text=cam_set.name,
                                    variable=cam_set.tk_var,
                                    command=self._auto_apply_bool_settings)

    cam_set.tk_obj.pack(anchor='w', side='top', expand=False, fill='none',
                        padx=5, pady=2)

  def _add_slider_setting(self, cam_set: CameraScaleSetting) -> None:
    """Adds a setting represented by a scale bar."""

    self.log(logging.DEBUG, f"Adding the slider setting {cam_set.name}")

    # The scale bar is slightly different if the setting type is int or float
    if cam_set.type == int:
      cam_set.tk_var = tk.IntVar(value=cam_set.value)
    else:
      cam_set.tk_var = tk.DoubleVar(value=cam_set.value)

    cam_set.tk_obj = tk.Scale(self._canvas_frame,
                              label=f'{cam_set.name} :',
                              variable=cam_set.tk_var,
                              resolution=cam_set.step,
                              orient='horizontal',
                              from_=cam_set.lowest,
                              to=cam_set.highest)

    cam_set.tk_obj.bind("<ButtonRelease-1>", self._auto_apply_scale_settings)

    cam_set.tk_obj.pack(anchor='center', side='top', expand=False,
                        fill='x', padx=5, pady=2)

  def _add_choice_setting(self, cam_set: CameraChoiceSetting) -> None:
    """Adds a setting represented by a list of radio buttons."""

    self.log(logging.DEBUG, f"Adding the choice setting {cam_set.name}")

    cam_set.tk_var = tk.StringVar(value=cam_set.value)
    label = tk.Label(self._canvas_frame, text=f'{cam_set.name} :')
    label.pack(anchor='w', side='top', expand=False, fill='none',
               padx=12, pady=2)

    for value in cam_set.choices:
      tk_obj = tk.Radiobutton(self._canvas_frame,
                              text=value,
                              variable=cam_set.tk_var,
                              value=value,
                              command=self._auto_apply_choice_settings)

      tk_obj.pack(anchor='w', side='top', expand=False,
                  fill='none', padx=5, pady=2)

      cam_set.tk_obj.append(tk_obj)

  def _set_variables(self) -> None:
    """Sets the text and numeric variables holding information about the
    display."""

    self.log(logging.DEBUG, "Setting the interface variables")

    # The FPS counter
    self._fps_var = tk.DoubleVar(value=0.)
    self._fps_txt = tk.StringVar(
        value=f'fps = {self._fps_var.get():.2f}\n(might be lower in this GUI '
              f'than actual)')

    # The variable for enabling or disabling the auto range
    self._auto_range = tk.BooleanVar(value=False)

    # The variable for enabling or disabling the auto apply
    self._auto_apply = tk.BooleanVar(value=False)

    # The minimum and maximum pixel value counters
    self._min_pixel = tk.IntVar(value=0)
    self._max_pixel = tk.IntVar(value=0)
    self._min_max_pix_txt = tk.StringVar(
      value=f'min: {self._min_pixel.get():d}, '
            f'max: {self._max_pixel.get():d}')

    # The number of detected bits counter
    self._nb_bits = tk.IntVar(value=0)
    self._bits_txt = tk.StringVar(
      value=f'Detected bits: {self._nb_bits.get():d}')

    # The display of the current zoom level
    self._zoom_level = tk.DoubleVar(value=100.0)
    self._zoom_txt = tk.StringVar(
      value=f'Zoom: {self._zoom_level.get():.1f}%')

    # The display of the current pixel position and value
    self._x_pos = tk.IntVar(value=0)
    self._y_pos = tk.IntVar(value=0)
    self._reticle_val = tk.IntVar(value=0)
    self._reticle_txt = tk.StringVar(value=f'X: {self._x_pos.get():d}, '
                                           f'Y: {self._y_pos.get():d}, '
                                           f'V: {self._reticle_val.get():d}')

  def _set_traces(self) -> None:
    """Sets the traces for automatically updating the display when a variable
    is modified."""

    self.log(logging.DEBUG, "Setting the interface traces")

    self._fps_var.trace_add('write', self._update_fps)

    self._min_pixel.trace_add('write', self._update_min_max)
    self._max_pixel.trace_add('write', self._update_min_max)

    self._nb_bits.trace_add('write', self._update_bits)

    self._zoom_level.trace_add('write', self._update_zoom)

    self._x_pos.trace_add('write', self._update_reticle)
    self._y_pos.trace_add('write', self._update_reticle)
    self._reticle_val.trace_add('write', self._update_reticle)

    self._auto_apply.trace_add('write', self._update_apply_settings)

  def _update_fps(self, _, __, ___) -> None:
    """Auto-update of the FPS display."""

    self._fps_txt.set(f'fps = {self._fps_var.get():.2f}\n'
                      f'(might be lower in this GUI than actual)')

  def _update_min_max(self, _, __, ___) -> None:
    """Auto-update of the minimum and maximum pixel values display."""

    self._min_max_pix_txt.set(f'min: {self._min_pixel.get():d}, '
                              f'max: {self._max_pixel.get():d}')

  def _update_bits(self, _, __, ___) -> None:
    """Auto-update of the number of detected bits display."""

    self._bits_txt.set(f'Detected bits: {self._nb_bits.get():d}')

  def _update_zoom(self, _, __, ___) -> None:
    """Auto-update of the current zoom level display."""

    self._zoom_txt.set(f'Zoom: {self._zoom_level.get():.1f}%')

  def _update_reticle(self, _, __, ___) -> None:
    """Auto-update of the current pixel position and value display."""

    self._reticle_txt.set(f'X: {self._x_pos.get():d}, '
                          f'Y: {self._y_pos.get():d}, '
                          f'V: {self._reticle_val.get():d}')

  def _update_apply_settings(self, _, __, ___) -> None:
    """Disable the Apply Settings button when Auto apply button is checked."""

    if self._auto_apply.get():
      self._update_button['state'] = 'disabled'
    else:
      self._update_button['state'] = 'normal'

  def _update_settings(self) -> None:
    """Tries to update the settings values upon clicking on the Apply Settings
    button, and checks that the settings have been correctly set."""

    for setting in self._camera.settings.values():
      # Applying all the settings that differ from the read value
      if setting.value != setting.tk_var.get():
        setting.value = setting.tk_var.get()

      # Reading the actual value of all the settings
      setting.tk_var.set(setting.value)

  def _auto_apply_scale_settings(self, _: tk.Event):
    """Applies the settings without clicking on the Apply Settings
     button when the Auto apply button is checked.

     The scale settings will be applied when the slicer is released.
     """

    if self._auto_apply.get():
      self._update_settings()

  def _auto_apply_bool_settings(self):
    """Applies the settings without clicking on the Apply Settings
     button when the Auto apply button is checked.

     The bool settings will be applied when the bool button is checked.
     """

    if self._auto_apply.get():
      self._update_settings()

  def _auto_apply_choice_settings(self):
    """Applies the settings without clicking on the Apply Settings
     button when the Auto apply button is checked.

     The choice settings will be applied when the choice button is checked.
     """

    if self._auto_apply.get():
      self._update_settings()

  def _cast_img(self, img: np.ndarray) -> None:
    """Casts the image to 8-bits as a greater precision is not required.

    May also interpolate the image to obtain a higher contrast, depending on
    the user's choice.
    """

    # First, convert BGR to RGB
    if len(img.shape) == 3:
      img = img[:, :, ::-1]

    # If the auto_range is set, adjusting the values to the range
    if self._auto_range.get():
      self.log(logging.DEBUG, "Applying auto range to the image")
      self._low_thresh, self._high_thresh = map(float,
                                                np.percentile(img, (3, 97)))
      self._img = ((np.clip(img, self._low_thresh, self._high_thresh) -
                    self._low_thresh) * 255 /
                   (self._high_thresh - self._low_thresh)).astype('uint8')

      # The original image still needs to be saved as 8-bits
      bit_depth = int(np.ceil(np.log2(int(np.max(img)) + 1)))
      self._original_img = (img / 2 ** (bit_depth - 8)).astype('uint8')

    # Or if the image is not already 8 bits, casting to 8 bits
    elif img.dtype != np.uint8:
      self.log(logging.DEBUG, "Casting the image to 8 bits")
      bit_depth = int(np.ceil(np.log2(int(np.max(img)) + 1)))
      self._img = (img / 2 ** (bit_depth - 8)).astype('uint8')
      self._original_img = np.copy(self._img)

    # Else, the image is usable as is
    else:
      self._img = img
      self._original_img = np.copy(img)

    # Updating the information
    self._nb_bits.set(int(np.ceil(np.log2(int(np.max(img)) + 1))))
    self._max_pixel.set(int(np.max(img)))
    self._min_pixel.set(int(np.min(img)))

  def _resize_img(self) -> None:
    """Resizes the received image so that it fits in the image canvas and
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

    # Creating the pillow image from the zoomed numpy array
    pil_img = Image.fromarray(zoomed_img)

    # Resizing the image to make it fit in the image canvas
    img_canvas_width = self._img_canvas.winfo_width()
    img_canvas_height = self._img_canvas.winfo_height()

    zoomed_img_ratio = pil_img.width / pil_img.height
    img_label_ratio = img_canvas_width / img_canvas_height

    if zoomed_img_ratio >= img_label_ratio:
      new_width = img_canvas_width
      new_height = max(int(img_canvas_width / zoomed_img_ratio), 1)
    else:
      new_width = max(int(img_canvas_height * zoomed_img_ratio), 1)
      new_height = img_canvas_height

    self._pil_img = pil_img.resize((new_width, new_height))

  def _display_img(self) -> None:
    """Displays the image in the center of the image canvas."""

    if self._pil_img is None:
      return

    self.log(logging.DEBUG, "Displaying the image")

    self._image_tk = ImageTk.PhotoImage(self._pil_img)
    self._img_canvas.create_image(int(self._img_canvas.winfo_width() / 2),
                                  int(self._img_canvas.winfo_height() / 2),
                                  anchor='center', image=self._image_tk)

  def _on_img_resize(self, _: Optional[tk.Event] = None) -> None:
    """Resizes the image and updates the display when the zoom level has
    changed or the GUI has been resized."""

    self.log(logging.DEBUG, "The image canvas was resized")

    self._draw_overlay()

    self._resize_img()
    self._display_img()
    self.update()

  def _calc_hist(self) -> None:
    """Calculates the histogram of the current image."""

    if self._original_img is None:
      return

    # Don't calculate histogram if a calculation is already running
    if self._processing_event.is_set():
      self.log(logging.DEBUG, "A calculation is running for the histogram, "
                              "not sending image for calculation")
      return

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
      self._img_in.put_nowait((hist_img, self._auto_range.get(),
                               self._low_thresh, self._high_thresh))

    # Checking if a histogram is available for display
    while not self._img_out.empty():
      self._hist = self._img_out.get_nowait()
      self.log(logging.DEBUG, "Received histogram from histogram process")

  def _resize_hist(self) -> None:
    """Resizes the histogram image to make it fit in the GUI."""

    if self._hist is None:
      return

    self.log(logging.DEBUG, "Resizing the histogram to fit in the window")

    pil_hist = Image.fromarray(self._hist)
    hist_canvas_width = self._hist_canvas.winfo_width()
    hist_canvas_height = self._hist_canvas.winfo_height()

    self._pil_hist = pil_hist.resize((hist_canvas_width, hist_canvas_height))

  def _display_hist(self) -> None:
    """Displays the histogram image in the GUI."""

    if self._pil_hist is None:
      return

    self.log(logging.DEBUG, "Displaying the histogram")

    self._hist_tk = ImageTk.PhotoImage(self._pil_hist)
    self._hist_canvas.create_image(int(self._hist_canvas.winfo_width() / 2),
                                   int(self._hist_canvas.winfo_height() / 2),
                                   anchor='center', image=self._hist_tk)

  def _on_hist_resize(self, _: tk.Event) -> None:
    """Resizes the histogram and updates the display when the GUI has been
    resized."""

    self._resize_hist()
    self._display_hist()
    self.update()

  def _update_img(self) -> None:
    """Acquires an image from the camera, casts and resizes it, calculates its
    histogram, displays them and updates the image information."""

    self.log(logging.DEBUG, "Updating the image")

    ret = self._camera.get_image()

    # Flag raised if no image could be grabbed
    no_img = ret is None

    # If no frame could be grabbed from the camera
    if no_img:
      # If it's the first call, generate error image to initialize the window
      if not self._got_first_img:
        self.log(logging.WARNING, "Could not get an image from the camera, "
                                  "displaying an error image instead")
        ret = None, np.array(Image.open(BytesIO(resource_string(
          'crappy', 'tool/data/no_image.png'))))
      # Otherwise, just pass
      else:
        self.log(logging.DEBUG, "No image returned by the camera")
        self.update()
        sleep(0.001)
        return

    # Always set, so that the error image is only ever loaded once
    self._got_first_img = True
    self._n_loops += 1
    _, img = ret

    if not no_img and img.dtype != self.dtype:
      self.dtype = img.dtype
    if not no_img and img.shape != self.shape:
      self.shape = img.shape

    self._cast_img(img)
    self._draw_overlay()
    self._resize_img()

    self._calc_hist()
    self._resize_hist()

    self._display_img()
    self._display_hist()

    self._update_pixel_value()

    self.update()

  def _draw_overlay(self) -> None:
    """Method meant to be used by subclasses for drawing an overlay on top of
    the image to display."""

    ...

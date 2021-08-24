# coding: utf-8

from ..._global import OptionalModule

try:
  import tkinter as tk
  from tkinter import ttk
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")
  ttk = OptionalModule("tkinter")


class FrameObjects(tk.Frame):
  """A very simple class that should be inherited by frames, to help create and
  maintain in order the file.

  How it works:
  - After frames defined, create a dict (or an orderedDict, for convenience),
  that will contain every widget.
  - Use appropriate method, and specify the frame and its dictionary. If
  variables are created, they will be a class attribute.

  These are commune args to specify to every widget:

  - widgets_dict: the dictionary in which the widget will be stored.
  - frame: the frame to put the widget.
  - name: the key to call the widget inside the dictionary.
  - text (if applicable): the text to show.
  """

  def __init__(self):
    super().__init__()

  def add_button(self, **kwargs):
    """To add a tkinter button.

    Args:
      text: the text to show inside the button.
      bg: background color.
      height, width: self-explanatory
      command_type: by default, clicking to the button executes the
      submit_command(command) method, with command as arg. To have different
      behavior, just specify "custom", or other string than "to_serial".
      command: the command to be executed, OR the string to pass to
      submit_command.
    """

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    name = kwargs.pop('name', 'Button')
    text = kwargs.pop('text', 'Button')
    bg = kwargs.pop('bg', 'white')
    height = kwargs.pop('height', 2)
    width = kwargs.pop('width', 10)
    command_type = kwargs.pop('command_type', 'to_serial')
    command = kwargs.pop('command', None)

    assert not kwargs, 'Error: unknown arg(s) in button definition:' + str(
      kwargs)

    if command_type != 'to_serial':
      widgets_dict[name] = tk.Button(frame,
                                     text=text,
                                     bg=bg,
                                     relief="raised",
                                     height=height, width=width,
                                     command=command,
                                     font=("Courier bold", 11))
    else:
      widgets_dict[name] = tk.Button(frame,
                                     text=text,
                                     bg=bg,
                                     relief="raised",
                                     height=height, width=width,
                                     command=lambda: self.submit_command(
                                       command),
                                     font=("Courier bold", 11))

  @staticmethod
  def add_label(**kwargs):
    """To add label.

    Args:
      font: to specify the text font, size, style.
      relief: to add some relief to the label.
    """

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    text = kwargs.pop('text', 'label')
    name = kwargs.pop('name', text)
    relief = kwargs.pop('relief', 'flat')
    font = kwargs.pop('font', ('Courier bold', 11))

    widgets_dict[name] = (tk.Label(frame,
                                   text=text,
                                   relief=relief,
                                   font=font))

  def add_entry(self, **kwargs):
    """To add an entry box. The adding of the entry box will add an attribute,
    if no variable is specified.

    Args:
      vartype: to specify which type the variable associated with the entry
        will be. Useful to make sure the user does not enter a string when a
        number is expected.
      variable: the variables name, which will be set as an attribute.
      width : the width of the entry box.
    """

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    entry_name = kwargs.pop('name', 'name')
    width = kwargs.pop('width', 10)
    vartype = kwargs.pop('vartype', tk.DoubleVar())
    variable = kwargs.pop("variable", None)

    if not variable:
      setattr(self, entry_name + '_var', vartype)
      widgets_dict[entry_name] = tk.Entry(frame,
                                          textvariable=getattr(self,
                                                               entry_name +
                                                               '_var'),
                                          width=width)
    else:
      widgets_dict[entry_name] = tk.Entry(frame,
                                          textvariable=variable,
                                          width=width)

  def add_checkbutton(self, **kwargs):
    """To add a checkbutton. Will create automatically a boolean attribute,
    which will represent the checkbutton state."""

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    text = kwargs.pop("text", None)
    name = kwargs.pop('name', 'checkbutton')
    variable_name = kwargs.pop('variable', name + '_var')

    var = tk.BooleanVar()

    setattr(self, variable_name, var)
    widgets_dict[name] = tk.Checkbutton(frame,
                                        text=text,
                                        variable=var)

  def add_combobox(self, **kwargs):
    """To add a combobox. Will automatically add an attribute.

    Args:
      entries: a list that contains every selectable option.
      variable: the name of the variable, that will become an attribute.
      default_index: to define which default entry to show on the combobox.
      var:
    """

    widgets_dict = kwargs.pop("widgets_dict", None)
    frame = kwargs.pop("frame", None)
    entries = kwargs.pop("entries", None)
    name = kwargs.pop("name", "combobox")
    variable_name = kwargs.pop("variable", name + "_var")
    default_index = kwargs.pop("default", 0)

    var = tk.StringVar()
    var.set(entries[default_index])

    setattr(self, variable_name, var)

    combo_box = ttk.Combobox(frame,
                             textvariable=var,
                             values=entries,
                             state='readonly')

    widgets_dict[name] = combo_box

  @staticmethod
  def add_scale(**kwargs):
    """To add a scrollbar"""

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    name = kwargs.pop('name', 'Button')
    boundaries = kwargs.pop("boundaries", (0, 1))

    widgets_dict[name] = tk.Scale(frame,
                                  from_=boundaries[0],
                                  to_=boundaries[1],
                                  orient=tk.HORIZONTAL,
                                  )

  @staticmethod
  def add_text(**kwargs):
    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    text = kwargs.pop('text', 'label')
    name = kwargs.pop('name', text)
    widgets_dict[name] = tk.Text(frame)

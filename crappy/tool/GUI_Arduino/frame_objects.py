# coding: utf-8
"""
A very simple class that should be inherited by frames, to help create and
maintain in order the file.
"""
import Tkinter as tk
import ttk


class FrameObjects(tk.Frame):
  def __init__(self):
    pass

  def add_button(self, **kwargs):

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    name = kwargs.pop('name', 'Button')
    text = kwargs.pop('text', 'Button')
    bg = kwargs.pop('bg', 'white')
    height = kwargs.pop('height', 2)
    width = kwargs.pop('width', 10)
    command = kwargs.pop('command', None)
    command_type = kwargs.pop('command_type', 'to_serial')

    assert not kwargs, 'Error: unknown arg(s) in button definition:' + str(
      kwargs)

    if command_type is not 'to_serial':
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

  def add_label(self, **kwargs):

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

    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    entry_name = kwargs.pop('entry_name', 'entry_name')
    width = kwargs.pop('width', 10)
    vartype = kwargs.pop('vartype', tk.DoubleVar())
    variable = kwargs.pop("variable", None)

    # Affect the variable associated with the entry to the self object.
    if not variable:
      setattr(self, entry_name + '_var', vartype)
      widgets_dict[entry_name] = tk.Entry(frame,
                                          textvariable=getattr(self, entry_name
                                                               + '_var'),
                                          width=width)
    else:
      widgets_dict[entry_name] = tk.Entry(frame,
                                          textvariable=variable,
                                          width=width)

  def add_checkbutton(self, **kwargs):
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

    widgets_dict = kwargs.pop("widgets_dict", None)
    frame = kwargs.pop("frame", None)
    entries = kwargs.pop("entries", None)
    name = kwargs.pop("name", "combobox")
    variable_name = kwargs.pop("variable", name + "_var")
    var = tk.StringVar()
    var.set(entries[0])
    setattr(self, variable_name, var)

    combo_box = ttk.Combobox(frame,
                             textvariable=var,
                             values=entries,
                             state='readonly')

    widgets_dict[name] = combo_box

  def add_scale(self, **kwargs):
    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    name = kwargs.pop('name', 'Button')
    boundaries = kwargs.pop("boundaries", (0, 1))

    widgets_dict[name] = tk.Scale(frame,
                                  from_=boundaries[0],
                                  to_=boundaries[1],
                                  orient=tk.HORIZONTAL,
                                  )

  def add_text(self, **kwargs):
    widgets_dict = kwargs.pop('widgets_dict', None)
    frame = kwargs.pop('frame', None)
    text = kwargs.pop('text', 'label')
    name = kwargs.pop('name', text)
    widgets_dict[name] = tk.Text(frame)

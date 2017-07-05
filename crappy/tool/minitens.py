# coding: utf-8
import Tkinter as tk
from collections import OrderedDict


class MinitensFrame(tk.Frame):
  def __init__(self, parent, **kwargs):
    """
    Frame to use with minitens in crappy.
    """
    tk.Frame.__init__(self, parent)
    self.grid()
    self.create_widgets(**kwargs)
    self.queue = kwargs.get("queue")

  def add_button(self, widgets_dict, frame, name='Button', text='Button',
                 bg='white', command=None, height=2, width=10):
    widgets_dict[name] = tk.Button(frame,
                                   text=text,
                                   bg=bg,
                                   relief="raised",
                                   height=height, width=width,
                                   command=lambda: self.submit_command(command),
                                   font=("Courier bold", 10))

  def add_label(self, widgets_dict, frame, name=None, text='label',
                relief='flat',
                font=("Courier bold", 11)):
    if not name:
      name = text
    widgets_dict[name] = (tk.Label(frame, text=text, relief=relief,
                                   font=font))

  def add_entry(self, widgets_dict, frame, entry_name, width=10):
    setattr(self, entry_name, tk.DoubleVar())
    widgets_dict[entry_name] = tk.Entry(frame,
                                        textvariable=getattr(self, entry_name),
                                        width=width)

  def add_listbox(self, widgets_dict, frame, name):
    widgets_dict[name] = tk.Listbox(frame)

  def add_checkbutton(self, widgets_dict, frame, name, variable):
    widgets_dict[name] = tk.Checkbutton(frame,
                                        text=None,
                                        variable=variable)

  def create_widgets(self, **kwargs):
    """
    Frames organization
      Frame_displayer: to display effort and displacement values.
      Frame_position: to start and stop the motor, according to pre-defined
      values.
      Frame_cycle: cycle generator.
    """

    self.create_displayer_menu()
    self.create_submit_menu()
    self.create_limits_menu()
    self.create_cycles_menu()

    self.frame_displayer.grid(row=0, column=0)
    self.frame_position.grid(row=1, column=0)
    self.frame_limits.grid(row=0, column=1)
    self.frame_cycles.grid(row=1, column=1)

  def create_displayer_menu(self):

    self.frame_displayer = tk.Frame(self,
                                    relief=tk.SUNKEN,
                                    borderwidth=1)

    self.frame_displayer_widgets = OrderedDict()

    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text="Effort",
                   name="Effort(N)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(self.frame_displayer_widgets,
                    self.frame_displayer,
                    name='tare_effort',
                    text='Zero',
                    command='tare_effort')

    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="effort")

    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='Position absolue: l',
                   name="Position(mm)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(self.frame_displayer_widgets,
                    self.frame_displayer,
                    name='tare_position',
                    text='Zero',
                    command='tare_position')
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='0.0',
                   name="position",
                   font=("Courier bold", 48))

    # self.add_label(self.frame_displayer_widgets,
    #                self.frame_displayer,
    #                text='l0',
    #                name="l0",
    #                font=("Courier bold", 10, "italic"))

    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="position_prct")

    self.add_entry(self.frame_displayer_widgets,
                   self.frame_displayer,
                   entry_name='l0_entry',
                   width=8)
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='Position relative: (l - l0) / l0',
                   name="Position(%)",
                   font=("Courier bold", 11, "bold"))

    self.l0_entry.set(1.0)

    self.frame_displayer_widgets["Effort(N)"].grid(row=0,
                                                   column=0,
                                                   columnspan=3,
                                                   sticky=tk.W)
    self.frame_displayer_widgets['tare_effort'].grid(row=1,
                                                     column=0,
                                                     sticky=tk.W)
    self.frame_displayer_widgets['effort'].grid(row=1,
                                                column=1,
                                                columnspan=2)

    self.frame_displayer_widgets['Position(mm)'].grid(row=2,
                                                      column=0,
                                                      columnspan=3,
                                                      sticky=tk.W)
    self.frame_displayer_widgets['tare_position'].grid(row=3,
                                                       column=0,
                                                       sticky=tk.W)
    self.frame_displayer_widgets['position'].grid(row=3,
                                                  column=1,
                                                  sticky=tk.W,
                                                  columnspan=2)

    self.frame_displayer_widgets['Position(%)'].grid(row=4, column=0,
                                                     sticky=tk.W,
                                                     columnspan=3)

    self.frame_displayer_widgets['position_prct'].grid(row=5,
                                                       column=1,
                                                       sticky=tk.W,
                                                       columnspan=2,
                                                       rowspan=3)

    # self.frame_displayer_widgets['l0'].grid(row=5, column=0,
    #                                         sticky=tk.W)
    self.frame_displayer_widgets['l0_entry'].grid(row=6, column=0,
                                                  sticky=tk.W)

  def create_submit_menu(self):

    self.frame_position = tk.Frame(self,
                                   relief=tk.SUNKEN,
                                   borderwidth=1)

    self.frame_position_widgets = OrderedDict()

    self.add_button(self.frame_position_widgets, self.frame_position,
                    text="Traction",
                    bg="white",
                    command="TRACTION",
                    name="TRACTION",
                    width=10, height=4)
    self.add_button(self.frame_position_widgets, self.frame_position,
                    text="STOP!",
                    bg="red",
                    command="STOP",
                    name="STOP",
                    width=10, height=4)
    self.add_button(self.frame_position_widgets, self.frame_position,
                    text="Compression",
                    bg="white",
                    command="COMPRESSION",
                    name="COMPRESSION",
                    width=10, height=4)

    for id, widget in enumerate(self.frame_position_widgets):
      self.frame_position_widgets[widget].grid(row=0 + 1, column=id)

  def create_limits_menu(self):

    self.frame_limits_widgets = OrderedDict()
    self.frame_limits = tk.Frame(self,
                                 relief=tk.SUNKEN,
                                 borderwidth=1)

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="Limites",
                   name="limites_title",
                   font=("Courier bold", 48))

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="Effort(N)",
                   name="Effort")

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="Position(mm)",
                   name="Position")

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="Position(%)",
                   name="Position_prct")

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="MAXIMUM",
                   name="Haute")

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="MINIMUM",
                   name="Basse")

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text="Actif?",
                   name="Actif")

    self.frame_limits_widgets['limites_title'].grid(row=0, columnspan=4)
    self.frame_limits_widgets['Effort'].grid(row=2, column=0, sticky=tk.W)
    self.frame_limits_widgets['Position'].grid(row=3, column=0, sticky=tk.W)
    self.frame_limits_widgets['Position_prct'].grid(row=4, column=0,
                                                    sticky=tk.W)
    self.frame_limits_widgets['Haute'].grid(row=1, column=1)
    self.frame_limits_widgets['Basse'].grid(row=1, column=2)
    self.frame_limits_widgets['Actif'].grid(row=1, column=3)

    labels_entries = ['lim_haute_effort',
                      'lim_basse_effort',
                      'lim_haute_position',
                      'lim_basse_position',
                      'lim_haute_position_prct',
                      'lim_basse_position_prct']

    for entry in labels_entries:
      self.add_entry(self.frame_limits_widgets,
                     self.frame_limits,
                     entry_name=entry)

    self.effort_lim_enabled = tk.IntVar()
    self.position_lim_enabled = tk.IntVar()
    self.position_prct_lim_enabled = tk.IntVar()

    self.effort_lim_enabled.set(0)
    self.position_lim_enabled.set(0)
    self.position_prct_lim_enabled.set(0)

    self.add_checkbutton(self.frame_limits_widgets,
                         self.frame_limits,
                         name="chck_effort",
                         variable=self.effort_lim_enabled)
    self.add_checkbutton(self.frame_limits_widgets,
                         self.frame_limits,
                         name="chck_position",
                         variable=self.position_lim_enabled)
    self.add_checkbutton(self.frame_limits_widgets,
                         self.frame_limits,
                         name="chck_position_prct",
                         variable=self.position_prct_lim_enabled)

    for id, widget in enumerate(labels_entries[::2]):
      self.frame_limits_widgets[widget].grid(row=2 + id, column=1)

    for id, widget in enumerate(labels_entries[1::2]):
      self.frame_limits_widgets[widget].grid(row=2 + id, column=2)

    self.frame_limits_widgets["chck_effort"].grid(row=2, column=3)
    self.frame_limits_widgets["chck_position"].grid(row=3, column=3)
    self.frame_limits_widgets["chck_position_prct"].grid(row=4, column=3)

  def create_cycles_menu(self):
    self.frame_cycles = tk.Frame(self,
                                 relief=tk.SUNKEN,
                                 borderwidth=1)

    self.frame_cycles_widgets = OrderedDict()
    self.add_label(self.frame_cycles_widgets,
                   self.frame_cycles,
                   text="CYCLES",
                   font=("Courier bold", 12))

    labels_cycles = [("Effort(N)", 'effort'),
                     ("Position(mm)", 'position'),
                     ("Position(%)", 'prct')]

    self.frame_cycles_widgets["cycles_type"] = tk.StringVar()
    self.frame_cycles_widgets["cycles_type"].set("position")
    self.cycles = []
    self.nb_cycles = 0.
    for label, mode in labels_cycles:
      self.frame_cycles_widgets[label] = tk.Radiobutton(self.frame_cycles,
                                                        text=label,
                                                        variable=
                                                        self.frame_cycles_widgets[
                                                          "cycles_type"],
                                                        value=mode)

    for limit in ["maximum", "minimum", "nombre"]:
      self.add_label(self.frame_cycles_widgets,
                     self.frame_cycles,
                     limit)
      self.add_entry(self.frame_cycles_widgets,
                     self.frame_cycles,
                     limit + '_entry')
    self.frame_cycles_widgets["submit_cycle"] = tk.Button(self.frame_cycles,
                                                          text="Soumettre!",
                                                          bg="white",
                                                          relief="raised",
                                                          height=2, width=10,
                                                          command=lambda: self.submit_cycle())

    self.frame_cycles_widgets["CYCLES"].grid(row=0, columnspan=4)
    self.frame_cycles_widgets["Effort(N)"].grid(row=1, column=0)
    self.frame_cycles_widgets["Position(mm)"].grid(row=1, column=1)
    self.frame_cycles_widgets["Position(%)"].grid(row=1, column=2)

    self.frame_cycles_widgets["maximum"].grid(row=2,
                                              column=0)  # , columnspan=1)
    self.frame_cycles_widgets["minimum"].grid(row=2,
                                              column=1)  # , columnspan=1)
    self.frame_cycles_widgets["nombre"].grid(row=2, column=2)
    self.frame_cycles_widgets["maximum_entry"].grid(row=3, column=0)
    self.frame_cycles_widgets["minimum_entry"].grid(row=3, column=1)
    self.frame_cycles_widgets["nombre_entry"].grid(row=3, column=2)
    self.frame_cycles_widgets["submit_cycle"].grid(row=3, column=3)

  def check_limits(self, effort, position, sens, prct):
    try:
      if self.position_lim_enabled.get():
        if position <= self.lim_basse_position.get() and sens == -1:
          self.submit_command("STOP")
        if position >= self.lim_haute_position.get() and sens == 1:
          self.submit_command("STOP")
      if self.effort_lim_enabled.get():
        if effort <= self.lim_basse_effort.get() and sens == -1:
          self.submit_command("STOP")
        if effort >= self.lim_haute_effort.get() and sens == 1:
          self.submit_command("STOP")
      if self.position_prct_lim_enabled.get():
        if prct <= self.lim_basse_position_prct.get() and sens == -1:
          self.submit_command("STOP")
        if prct >= self.lim_haute_position_prct.get() and sens == 1:
          self.submit_command("STOP")
    except ValueError:
      pass

  def submit_cycle(self):

    nb_cycles = int(self.frame_cycles_widgets["nombre_entry"].get())
    self.frame_cycles_widgets["nombre_entry"].delete(0, tk.END)
    for i in range(nb_cycles):
      new_cycle = {}
      for entry in ["maximum_entry", "minimum_entry"]:
        new_cycle[entry] = self.frame_cycles_widgets[entry].get()
      new_cycle["cycles_type"] = self.frame_cycles_widgets["cycles_type"].get()
      self.cycles.append(new_cycle)
      print('cycles', self.cycles)

    self.frame_cycles_widgets["maximum_entry"].delete(0, tk.END)
    self.frame_cycles_widgets["minimum_entry"].delete(0, tk.END)

  def check_cycle(self, eff, sns, pos, prct):
    if self.cycles[0]["cycles_type"] == "position":
      var = pos
    elif self.cycles[0]["cycles_type"] == "effort":
      var = eff
    elif self.cycles[0]["cycles_type"] == "prct":
      var = prct
    print(self.cycles)
    if var >= float(self.cycles[0]["maximum_entry"]) and sns == 1:
      self.nb_cycles += 0.5
      self.submit_command("COMPRESSION")
    if var <= float(self.cycles[0]["minimum_entry"]) and sns == -1:
      self.nb_cycles += 0.5
      del self.cycles[0]
      if self.cycles:
        self.submit_command("TRACTION")
      else:
        self.submit_command("STOP")

  def submit_command(self, arg):

    if arg == "STOP":
      dico = {"sns": 0}
    elif arg == "TRACTION":
      dico = {"sns": 1}
    elif arg == "COMPRESSION":
      dico = {"sns": -1}
    elif arg == "tare_position":
      dico = {"t": 1}
    elif arg == "tare_effort":
      dico = {"t": 2}
    message = str(dico)
    self.queue.put(message)

  def update_data(self, message):

    try:
      eff = message.get('eff')
      sns = message.get('sns')
      mil = message.get('mil')
      pos = message.get('pos') / (60. * 1e3)
    except (TypeError, AttributeError):
      pos = 0.0
      eff = 0.0
      sns = 0.0
      mil = 0.0
    dico = {"position_abs": pos, "effort": eff, "sens": sns}
    try:
      prct = (pos - self.l0_entry.get()) / float(self.l0_entry.get())
      dico["position_prct"] = prct
    except (ValueError, ZeroDivisionError):
      prct = 0.
    self.check_limits(eff, pos, sns, prct)
    if self.cycles:
      self.check_cycle(eff, sns, pos, prct)
    self.update_widgets(dico)

  def update_widgets(self, message):

    mot = format(message["position_abs"], '.5f') + ' mm'
    self.frame_displayer_widgets["position"].configure(text=mot)

    mot = format(message["effort"],'.5f') + ' N'
    self.frame_displayer_widgets["effort"].configure(text=mot)

    if "position_prct" in message:
      mot = format(message["position_prct"], '.5f') + ' %'
      self.frame_displayer_widgets["position_prct"].configure(text=mot)

    if message["sens"] == 1:
      self.frame_position_widgets["TRACTION"].configure(bg="blue")
    elif message["sens"] == -1:
      self.frame_position_widgets["COMPRESSION"].configure(bg="blue")
    else:
      self.frame_position_widgets["COMPRESSION"].configure(bg="white")
      self.frame_position_widgets["TRACTION"].configure(bg="white")

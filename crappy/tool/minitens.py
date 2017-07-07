# coding: utf-8
import Tkinter as tk
import ttk
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

  def add_button(self, **kwargs):

    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    name = kwargs.get('name', 'Button')
    text = kwargs.get('text', 'Button')
    bg = kwargs.get('bg', 'white')
    height = kwargs.get('height', 2)
    width = kwargs.get('width', 10)
    command = kwargs.get('command', None)
    command_type = kwargs.get('command_type', 'to_serial')

    if command_type is not 'to_serial':
      widgets_dict[name] = tk.Button(frame,
                                     text=text,
                                     bg=bg,
                                     relief="raised",
                                     height=height, width=width,
                                     command=command,
                                     font=("Courier bold", 10))
    else:
      widgets_dict[name] = tk.Button(frame,
                                     text=text,
                                     bg=bg,
                                     relief="raised",
                                     height=height, width=width,
                                     command=lambda: self.submit_command(
                                       command),
                                     font=("Courier bold", 10))

  def add_label(self, **kwargs):

    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    text = kwargs.get('text', 'label')
    name = kwargs.get('name', text)
    relief = kwargs.get('relief', 'flat')
    font = kwargs.get('font', ('Courier bold', 11))

    widgets_dict[name] = (tk.Label(frame,
                                   text=text,
                                   relief=relief,
                                   font=font))

  def add_entry(self, **kwargs):

    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    entry_name = kwargs.get('entry_name', 'entry_name')
    width = kwargs.get('width', 10)
    vartype = kwargs.get('vartype', tk.DoubleVar())

    # Affect the variable associated with the entry to the self object.
    setattr(self, entry_name, vartype)

    widgets_dict[entry_name] = tk.Entry(frame,
                                        textvariable=getattr(self, entry_name),
                                        width=width)

  def add_listbox(self, **kwargs):

    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    name = kwargs.get('name', 'listbox')

    widgets_dict[name] = tk.Listbox(frame)

  def add_checkbutton(self, **kwargs):
    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    name = kwargs.get('name', 'checkbutton')
    variable = kwargs.get('variable', 'variable')

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

    self.create_menubar()
    self.create_displayer_menu()
    self.create_submit_menu()
    # self.create_limits_menu()
    self.create_cycles_menu()
    self.create_popup_limits()

    self.frame_displayer.grid(row=0, column=0, rowspan=2)
    self.frame_position.grid(row=1, column=1)
    # self.frame_limits.grid(row=0, column=1)
    self.frame_cycles.grid(row=2, column=0, columnspan=2)

  def create_menubar(self):
    self.menubar = tk.Menu(self)
    self.menubar.add_command(label='Limites',
                             command=lambda: self.create_popup_limits())
    self.menubar.add_command(label='Vitesse')
    self.menubar.add_command(label='Paramètres échantillon')
    self.menubar.add_command(label='Consigne...')

  def create_popup_limits(self):
    self.popup_limits = tk.Toplevel()
    self.popup_limits.title("Limites")
    self.popup_limits_menubar = tk.Menu(self.popup_limits)
    self.popup_limits_menubar.add_command(label="Aide",
                                          command=lambda: tk.Message("oui"))

    self.popup_limits.config(menu=self.popup_limits_menubar)
    self.create_limits_menu()

  def create_displayer_menu(self):

    self.frame_displayer = tk.Frame(self,
                                    relief=tk.SUNKEN,
                                    borderwidth=1)

    self.frame_displayer_widgets = OrderedDict()

    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   text="Effort",
                   name="Effort(N)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(widgets_dict=self.frame_displayer_widgets,
                    frame=self.frame_displayer,
                    name='tare_effort',
                    text='Zero',
                    command='tare_effort')

    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="effort")

    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   text='Position absolue: l',
                   name="Position(mm)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(widgets_dict=self.frame_displayer_widgets,
                    frame=self.frame_displayer,
                    name='tare_position',
                    text='Zero',
                    command='tare_position')

    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   name="position",
                   font=("Courier bold", 48))

    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="position_prct")

    self.add_entry(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
                   entry_name='l0_entry',
                   width=8)
    self.add_label(widgets_dict=self.frame_displayer_widgets,
                   frame=self.frame_displayer,
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

    self.add_button(widgets_dict=self.frame_position_widgets,
                    frame=self.frame_position,
                    text="Traction",
                    bg="white",
                    command="TRACTION",
                    name="TRACTION",
                    width=10, height=4)
    self.add_button(widgets_dict=self.frame_position_widgets,
                    frame=self.frame_position,
                    text="STOP!",
                    bg="red",
                    command="STOP",
                    name="STOP",
                    width=10, height=4)
    self.add_button(widgets_dict=self.frame_position_widgets,
                    frame=self.frame_position,
                    text="Compression",
                    bg="white",
                    command="COMPRESSION",
                    name="COMPRESSION",
                    width=10, height=4)

    for id, widget in enumerate(self.frame_position_widgets):
      self.frame_position_widgets[widget].grid(row=0 + 1, column=id)

  def create_limits_menu(self):

    self.frame_limits_widgets = OrderedDict()

    self.frame_limits = tk.Frame(self.popup_limits,
                                 relief=tk.SUNKEN,
                                 borderwidth=1)

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="Limites",
                   name="limites_title",
                   font=("Courier bold", 14, "bold"))

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="Effort(N)",
                   name="Effort")

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="Position(mm)",
                   name="Position")

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="Position(%)",
                   name="Position_prct")

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="MAXIMUM",
                   name="Haute")

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
                   text="MINIMUM",
                   name="Basse")

    self.add_label(widgets_dict=self.frame_limits_widgets,
                   frame=self.frame_limits,
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
      self.add_entry(widgets_dict=self.frame_limits_widgets,
                     frame=self.frame_limits,
                     entry_name=entry)

    self.effort_lim_enabled = tk.IntVar()
    self.position_lim_enabled = tk.IntVar()
    self.position_prct_lim_enabled = tk.IntVar()

    self.effort_lim_enabled.set(0)
    self.position_lim_enabled.set(0)
    self.position_prct_lim_enabled.set(0)

    self.add_checkbutton(widgets_dict=self.frame_limits_widgets,
                         frame=self.frame_limits,
                         name="chck_effort",
                         variable=self.effort_lim_enabled)

    self.add_checkbutton(widgets_dict=self.frame_limits_widgets,
                         frame=self.frame_limits,
                         name="chck_position",
                         variable=self.position_lim_enabled)

    self.add_checkbutton(widgets_dict=self.frame_limits_widgets,
                         frame=self.frame_limits,
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
    self.add_label(widgets_dict=self.frame_cycles_widgets,
                   frame=self.frame_cycles,
                   text="GENERATEUR DE CYCLES",
                   name="CYCLES",
                   font=("Courier bold", 14, "bold"))

    self.frame_cycles_widgets["cycles_type"] = tk.StringVar()
    self.frame_cycles_widgets["cycles_type"].set("position")
    self.cycles = []
    self.nb_cycles = 0.
    self.cycles_started = False
    self.rising = True

    # for label, mode in labels_cycles:
    #   self.frame_cycles_widgets[label] = tk.Radiobutton(self.frame_cycles,
    #                                                     text=label,
    #                                                     variable=
    #                                                     self.frame_cycles_widgets[
    #                                                       "cycles_type"],
    #                                                     value=mode)
    #
    # for limit in ["maximum", "minimum"]:
    #   self.add_label(widgets_dict=self.frame_cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  name=limit,
    #                  text=limit)
    #   self.add_entry(widgets_dict=self.frame_cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  entry_name=limit + '_entry')
    # self.add_label(widgets_dict=self.frame_cycles_widgets,
    #                frame=self.frame_cycles,
    #                name="nombre",
    #                text="nombre")
    # self.add_entry(widgets_dict=self.frame_cycles_widgets,
    #                frame=self.frame_cycles,
    #                entry_name="nombre_entry",
    #                vartype=tk.IntVar())

    self.add_button(widgets_dict=self.frame_cycles_widgets,
                    frame=self.frame_cycles,
                    name="submit_cycle",
                    text="Soumettre \n nouveau cycle...",
                    bg="white",
                    command_type="custom",
                    command=lambda: self.submit_new_cycle(),
                    width=15, height=5)

    self.add_button(widgets_dict=self.frame_cycles_widgets,
                    frame=self.frame_cycles,
                    name="start_cycle",
                    text="Démarrer",
                    bg="green",
                    command_type="custom",
                    command=lambda: self.start_cycle(),
                    width=15, height=5)

    columns = ("Type de limite haute",
               "Limite haute",
               "Type de limite basse",
               "Limite basse")

    self.cycles_table = ttk.Treeview(self.frame_cycles)
    self.cycles_table["columns"] = columns

    self.cycles_table.heading('#0', text="#")
    self.cycles_table.column('#0', width=50)

    for column in columns:
      self.cycles_table.column(column, width=20 * len(column))
      self.cycles_table.heading(column, text=column)

    self.frame_cycles_widgets["CYCLES"].grid(row=0, columnspan=4)
    self.frame_cycles_widgets["submit_cycle"].grid(row=1, column=0,
                                                   columnspan=2)

    self.frame_cycles_widgets["start_cycle"].grid(row=1, column=2,
                                                  columnspan=2)
    self.cycles_table.grid(row=5, column=0, columnspan=4, rowspan=1)

  def add_combobox(self, **kwargs):

    widgets_dict = kwargs.get("widgets_dict", None)
    frame = kwargs.get("frame", None)
    entries = kwargs.get("entries", None)
    name = kwargs.get("name", "combobox")
    variable = kwargs.get("variable", None)

    widgets_dict[variable] = tk.StringVar()

    combo_box = ttk.Combobox(frame,
                             textvariable=widgets_dict[variable],
                             values=entries,
                             state='readonly')

    widgets_dict[name + '_combobox'] = combo_box

  def submit_new_cycle(self):

    if hasattr(self, 'cycles_popup'):
      return

    self.cycles_popup = tk.Toplevel()
    self.cycles_popup.title("Nouveau cycle")

    entries_combobox = ('Position(%)', 'Position(mm)', 'Effort(N)')

    labels_cycles = (("max_type_label", "Type de valeur max"),
                     ("max_label", "Valeur max"),
                     ("min_type_label", "Type de valeur min"),
                     ("min_label", "Valeur min"),
                     ("nb_label", "Nombre"))

    for name, label in labels_cycles:
      self.add_label(widgets_dict=self.frame_cycles_widgets,
                     frame=self.cycles_popup,
                     name=name,
                     text=label)

    self.add_entry(widgets_dict=self.frame_cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="maximum_entry")

    self.add_combobox(widgets_dict=self.frame_cycles_widgets,
                      frame=self.cycles_popup,
                      name="maximum_type",
                      entries=entries_combobox,
                      variable="maximum_type")

    self.add_entry(widgets_dict=self.frame_cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="minimum_entry")

    self.add_combobox(widgets_dict=self.frame_cycles_widgets,
                      frame=self.cycles_popup,
                      name="minimum_type",
                      entries=entries_combobox,
                      variable="minimum_type")

    self.add_entry(widgets_dict=self.frame_cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="nombre_entry")

    self.add_button(widgets_dict=self.frame_cycles_widgets,
                    frame=self.cycles_popup,
                    name="submit",
                    text="Soumettre",
                    command=lambda: self.submit_cycle(),
                    command_type='custom')

    self.add_button(widgets_dict=self.frame_cycles_widgets,
                    frame=self.cycles_popup,
                    text="Terminer",
                    name="quit",
                    command=lambda: self.close_cycles_popup(),
                    command_type='custom')

    for index, (name, _) in enumerate(labels_cycles):
      self.frame_cycles_widgets[name].grid(row=0, column=index)

    self.frame_cycles_widgets["maximum_type_combobox"].grid(row=1, column=0)
    self.frame_cycles_widgets["maximum_entry"].grid(row=1, column=1)
    self.frame_cycles_widgets["minimum_type_combobox"].grid(row=1, column=2)
    self.frame_cycles_widgets["minimum_entry"].grid(row=1, column=3)
    self.frame_cycles_widgets["nombre_entry"].grid(row=1, column=4)
    self.frame_cycles_widgets["submit"].grid(row=2, column=1, columnspan=2)
    self.frame_cycles_widgets["quit"].grid(row=2, column=3, columnspan=2)

  def close_cycles_popup(self):
    self.cycles_popup.destroy()
    del self.cycles_popup

    # for limit in ["maximum", "minimum"]:
    #   self.add_label(widgets_dict=self.frame_cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  name=limit,
    #                  text=limit)
    #   self.add_entry(widgets_dict=self.frame_cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  entry_name=limit + '_entry')
    # self.add_label(widgets_dict=self.frame_cycles_widgets,
    #                frame=self.frame_cycles,
    #                name="nombre",
    #                text="nombre")
    # self.add_entry(widgets_dict=self.frame_cycles_widgets,
    #                frame=self.frame_cycles,
    #                entry_name="nombre_entry",
    #                vartype=tk.IntVar())

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
    try:
      nb_cycles = int(self.nombre_entry.get())

      for i in xrange(nb_cycles):
        new_cycle = [len(self.cycles) + 1,
                     self.frame_cycles_widgets["maximum_type"].get(),
                     self.maximum_entry.get(),
                     self.frame_cycles_widgets["minimum_type"].get(),
                     self.minimum_entry.get()]
        self.cycles.append(new_cycle)
        self.cycles_table.insert("", "end",
                                 text=new_cycle[0],
                                 values=new_cycle[1:])

    except ValueError:
      pass
    self.maximum_entry.set(0.0)
    self.minimum_entry.set(0.0)
    self.nombre_entry.set(0)

  def start_cycle(self):
    if self.cycles:
      self.cycles_started = True
      self.submit_command("TRACTION")

  def set_current_cycle(self, pos, eff, prct):
    pass

  def check_cycle(self, eff, sns, pos, prct):

    mode_max, maximum, mode_min, minimum = self.cycles[0][1:]

    if self.rising:
      if mode_max == "Effort(N)":
        var = eff
      elif mode_max == "Position(%)":
        var = prct
      elif mode_max == "Position(mm)":
        var = pos
    else:
      if mode_min == "Effort(N)":
        var = eff
      elif mode_min == "Position(%)":
        var = prct
      elif mode_min == "Position(mm)":
        var = pos


    if var >= maximum and sns == 1 and self.rising:
      self.nb_cycles += 0.5
      self.rising = False
      self.submit_command("COMPRESSION")

    elif var <= minimum and sns == -1 and not self.rising:

      self.nb_cycles += 0.5
      self.rising = True
      del self.cycles[0]
      self.cycles_table.delete(self.cycles_table.get_children()[0])
      if self.cycles:
        self.submit_command("TRACTION")
      else:
        self.cycles_started = False
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
    if self.cycles_started:
      self.check_cycle(eff, sns, pos, prct)
    self.update_widgets(dico)

  def update_widgets(self, message):

    mot = format(message["position_abs"], '.5f') + ' mm'
    self.frame_displayer_widgets["position"].configure(text=mot)

    mot = format(message["effort"], '.5f') + ' N'
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

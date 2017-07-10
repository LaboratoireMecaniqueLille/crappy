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

    self.frames = ["displayer",
                   "action",
                   "cycles"]
    for frame in self.frames:
      setattr(self, "frame_" + frame, tk.Frame(self,
                                               relief=tk.SUNKEN,
                                               borderwidth=1))
      setattr(self, frame + "_widgets", OrderedDict())

    self.create_widgets(**kwargs)
    self.queue = kwargs.get("queue")
    self.submit_command({"vts": 100})

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
    self.create_frame_display()
    self.create_frame_action()
    # self.create_menu_limits()
    self.create_frame_cycles()
    # self.create_popup_limits()

    self.frame_displayer.grid(row=0, column=0, rowspan=2)
    self.frame_action.grid(row=1, column=1)
    # self.frame_limits.grid(row=0, column=1)
    self.frame_cycles.grid(row=2, column=0, columnspan=2)

  def create_menubar(self):
    self.menubar = tk.Menu(self)
    self.menubar.add_command(label='Limites',
                             command=self.create_popup_limits)
    self.menubar.add_command(label='Vitesse',
                             command=self.create_popup_speed)
    self.menubar.add_command(label='Paramètres échantillon',
                             command=self.create_popup_sample_parameters)
    self.menubar.add_command(label='Consigne...',
                             command=self.create_popup_command)

  def create_popup_limits(self):

    if hasattr(self, 'popup_limits'):
      self.popup_limits.deiconify()
      return

    self.popup_limits = tk.Toplevel()
    self.popup_limits.protocol("WM_DELETE_WINDOW",
                               self.popup_limits.withdraw)

    self.popup_limits.title("Limites")
    self.popup_limits_menubar = tk.Menu(self.popup_limits)
    self.popup_limits_menubar.add_command(label="Aide",
                                          command=lambda: None)

    self.popup_limits.config(menu=self.popup_limits_menubar)
    self.create_menu_limits(self.popup_limits)

  def create_frame_display(self):

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text="Effort",
                   name="Effort(N)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(widgets_dict=self.displayer_widgets,
                    frame=self.frame_displayer,
                    name='tare_effort',
                    text='Zero',
                    command='tare_effort')

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="effort")

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text='Position absolue: l',
                   name="Position(mm)",
                   font=("Courier bold", 11, "bold"))

    self.add_button(widgets_dict=self.displayer_widgets,
                    frame=self.frame_displayer,
                    name='tare_position',
                    text='Zero',
                    command='tare_position')

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   name="position",
                   font=("Courier bold", 48))

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text='0.0',
                   font=("Courier bold", 48),
                   name="position_prct")

    # self.add_entry(widgets_dict=self.displayer_widgets,
    #                frame=self.frame_displayer,
    #                entry_name='l0_entry',
    #                width=8)

    self.add_label(widgets_dict=self.displayer_widgets,
                   frame=self.frame_displayer,
                   text='Position relative: (l - l0) / l0',
                   name="Position(%)",
                   font=("Courier bold", 11, "bold"))

    # self.l0_entry.set(1.0)

    self.displayer_widgets["Effort(N)"].grid(row=0,
                                             column=0,
                                             columnspan=3,
                                             sticky=tk.W)
    self.displayer_widgets['tare_effort'].grid(row=1,
                                               column=0,
                                               sticky=tk.W)
    self.displayer_widgets['effort'].grid(row=1,
                                          column=1,
                                          columnspan=2)

    self.displayer_widgets['Position(mm)'].grid(row=2,
                                                column=0,
                                                columnspan=3,
                                                sticky=tk.W)
    self.displayer_widgets['tare_position'].grid(row=3,
                                                 column=0,
                                                 sticky=tk.W)
    self.displayer_widgets['position'].grid(row=3,
                                            column=1,
                                            sticky=tk.W,
                                            columnspan=2)

    # self.displayer_widgets['Position(%)'].grid(row=4, column=0,
    #                                            sticky=tk.W,
    #                                            columnspan=3)
    #
    # self.displayer_widgets['position_prct'].grid(row=5,
    #                                              column=1,
    #                                              sticky=tk.W,
    #                                              columnspan=2,
    #                                              rowspan=3)

    # self.displayer_widgets['l0'].grid(row=5, column=0,
    #                                         sticky=tk.W)
    # self.displayer_widgets['l0_entry'].grid(row=6, column=0,
    #                                         sticky=tk.W)

  def create_frame_action(self):

    self.add_button(widgets_dict=self.action_widgets,
                    frame=self.frame_action,
                    text="Traction",
                    bg="white",
                    command="TRACTION",
                    name="TRACTION",
                    width=10, height=4)
    self.add_button(widgets_dict=self.action_widgets,
                    frame=self.frame_action,
                    text="STOP!",
                    bg="red",
                    command="STOP",
                    name="STOP",
                    width=10, height=4)
    self.add_button(widgets_dict=self.action_widgets,
                    frame=self.frame_action,
                    text="Compression",
                    bg="white",
                    command="COMPRESSION",
                    name="COMPRESSION",
                    width=10, height=4)

    for id, widget in enumerate(self.action_widgets):
      self.action_widgets[widget].grid(row=0 + 1, column=id)

  def create_menu_limits(self, frame):

    self.limits_widgets = OrderedDict()
    self.frame_limits = frame

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Limites",
                   name="limites_title",
                   font=("Courier bold", 14, "bold"))

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Effort(N)",
                   name="Effort")

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Position(mm)",
                   name="Position")

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Position(%)",
                   name="Position_prct")

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="MAXIMUM",
                   name="Haute")

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="MINIMUM",
                   name="Basse")

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Actif?",
                   name="Actif")

    self.limits_widgets['limites_title'].grid(row=0, columnspan=4)
    self.limits_widgets['Effort'].grid(row=2, column=0, sticky=tk.W)
    self.limits_widgets['Position'].grid(row=3, column=0, sticky=tk.W)

    self.limits_widgets['Position_prct'].grid(row=4, column=0,
                                              sticky=tk.W)
    self.limits_widgets['Haute'].grid(row=1, column=1)
    self.limits_widgets['Basse'].grid(row=1, column=2)
    self.limits_widgets['Actif'].grid(row=1, column=3)

    labels_entries = ['lim_haute_effort',
                      'lim_basse_effort',
                      'lim_haute_position',
                      'lim_basse_position',
                      'lim_haute_position_prct',
                      'lim_basse_position_prct']

    for entry in labels_entries:
      self.add_entry(widgets_dict=self.limits_widgets,
                     frame=self.frame_limits,
                     entry_name=entry)

    self.effort_lim_enabled = tk.IntVar()
    self.position_lim_enabled = tk.IntVar()
    self.position_prct_lim_enabled = tk.IntVar()

    self.effort_lim_enabled.set(0)
    self.position_lim_enabled.set(0)
    self.position_prct_lim_enabled.set(0)

    self.add_checkbutton(widgets_dict=self.limits_widgets,
                         frame=self.frame_limits,
                         name="chck_effort",
                         variable=self.effort_lim_enabled)

    self.add_checkbutton(widgets_dict=self.limits_widgets,
                         frame=self.frame_limits,
                         name="chck_position",
                         variable=self.position_lim_enabled)

    self.add_checkbutton(widgets_dict=self.limits_widgets,
                         frame=self.frame_limits,
                         name="chck_position_prct",
                         variable=self.position_prct_lim_enabled)

    for id, widget in enumerate(labels_entries[::2]):
      self.limits_widgets[widget].grid(row=2 + id, column=1)

    for id, widget in enumerate(labels_entries[1::2]):
      self.limits_widgets[widget].grid(row=2 + id, column=2)

    self.limits_widgets["chck_effort"].grid(row=2, column=3)
    self.limits_widgets["chck_position"].grid(row=3, column=3)
    self.limits_widgets["chck_position_prct"].grid(row=4, column=3)

  def create_frame_cycles(self):

    # self.frame_cycles = tk.Frame(self,
    #                              relief=tk.SUNKEN,
    #                              borderwidth=1)
    #
    # self.cycles_widgets = OrderedDict()
    self.add_label(widgets_dict=self.cycles_widgets,
                   frame=self.frame_cycles,
                   text="GENERATEUR DE CYCLES",
                   name="CYCLES",
                   font=("Courier bold", 14, "bold"))

    self.cycles_widgets["cycles_type"] = tk.StringVar()
    self.cycles_widgets["cycles_type"].set("position")
    self.cycles = []
    self.nb_cycles = 0.
    self.cycles_started = False
    self.rising = True

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.frame_cycles,
                    name="submit_cycle",
                    text="Soumettre \n nouveaux cycles...",
                    bg="white",
                    command_type="custom",
                    command=self.create_popup_new_cycle,
                    width=15, height=5)

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.frame_cycles,
                    name="start_cycle",
                    text="Démarrer",
                    bg="green",
                    command_type="custom",
                    command=self.start_cycle,
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
      self.cycles_table.column(column, width=10 * len(column))
      self.cycles_table.heading(column, text=column)

    self.cycles_widgets["CYCLES"].grid(row=0, columnspan=4)
    self.cycles_widgets["submit_cycle"].grid(row=1, column=0,
                                             columnspan=2)

    self.cycles_widgets["start_cycle"].grid(row=1, column=2,
                                            columnspan=2)
    self.cycles_table.grid(row=5, column=0, columnspan=4, rowspan=1)

  def add_combobox(self, **kwargs):

    widgets_dict = kwargs.get("widgets_dict", None)
    frame = kwargs.get("frame", None)
    entries = kwargs.get("entries", None)
    name = kwargs.get("name", "combobox")
    variable = kwargs.get("variable", None)

    widgets_dict[variable] = tk.StringVar()
    widgets_dict[variable].set(entries[0])
    combo_box = ttk.Combobox(frame,
                             textvariable=widgets_dict[variable],
                             values=entries,
                             state='readonly')

    widgets_dict[name + '_combobox'] = combo_box

  def create_popup_new_cycle(self):

    if hasattr(self, 'cycles_popup'):
      self.cycles_popup.deiconify()
      return

    self.cycles_popup = tk.Toplevel()
    self.cycles_popup.title("Nouveau cycle")
    self.cycles_popup.protocol("WM_DELETE_WINDOW",
                               self.cycles_popup.withdraw)

    entries_combobox = ('Position(%)', 'Position(mm)', 'Effort(N)')

    labels_cycles = (("max_type_label", "Type de valeur max"),
                     ("max_label", "Valeur max"),
                     ("min_type_label", "Type de valeur min"),
                     ("min_label", "Valeur min"),
                     ("nb_label", "Nombre"))

    for name, label in labels_cycles:
      self.add_label(widgets_dict=self.cycles_widgets,
                     frame=self.cycles_popup,
                     name=name,
                     text=label)

    self.add_entry(widgets_dict=self.cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="maximum_entry")

    self.add_combobox(widgets_dict=self.cycles_widgets,
                      frame=self.cycles_popup,
                      name="maximum_type",
                      entries=entries_combobox,
                      variable="maximum_type")

    self.add_entry(widgets_dict=self.cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="minimum_entry")

    self.add_combobox(widgets_dict=self.cycles_widgets,
                      frame=self.cycles_popup,
                      name="minimum_type",
                      entries=entries_combobox,
                      variable="minimum_type")

    self.add_entry(widgets_dict=self.cycles_widgets,
                   frame=self.cycles_popup,
                   entry_name="nombre_entry")

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.cycles_popup,
                    name="submit",
                    text="Soumettre",
                    command=lambda: self.submit_cycle(),
                    command_type='custom')

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.cycles_popup,
                    text="Terminer",
                    name="quit",
                    command=self.cycles_popup.withdraw,
                    command_type='custom')

    for index, (name, _) in enumerate(labels_cycles):
      self.cycles_widgets[name].grid(row=0, column=index)

    self.cycles_widgets["maximum_type_combobox"].grid(row=1, column=0)
    self.cycles_widgets["maximum_entry"].grid(row=1, column=1)
    self.cycles_widgets["minimum_type_combobox"].grid(row=1, column=2)
    self.cycles_widgets["minimum_entry"].grid(row=1, column=3)
    self.cycles_widgets["nombre_entry"].grid(row=1, column=4)
    self.cycles_widgets["submit"].grid(row=2, column=1, columnspan=2)
    self.cycles_widgets["quit"].grid(row=2, column=3, columnspan=2)

    # def close_cycles_popup(self):
    # self.cycles_popup.destroy()
    # del self.cycles_popup

    # for limit in ["maximum", "minimum"]:
    #   self.add_label(widgets_dict=self.cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  name=limit,
    #                  text=limit)
    #   self.add_entry(widgets_dict=self.cycles_widgets,
    #                  frame=self.frame_cycles,
    #                  entry_name=limit + '_entry')
    # self.add_label(widgets_dict=self.cycles_widgets,
    #                frame=self.frame_cycles,
    #                name="nombre",
    #                text="nombre")
    # self.add_entry(widgets_dict=self.cycles_widgets,
    #                frame=self.frame_cycles,
    #                entry_name="nombre_entry",
    #                vartype=tk.IntVar())

  def create_popup_sample_parameters(self):
    if hasattr(self, 'sample_parameters'):
      self.sample_parameters.deiconify()
      return

    self.sample_parameters = tk.Toplevel()
    self.sample_parameters.title("en mm")
    self.sample_parameters.protocol("WM_DELETE_WINDOW",
                                    self.sample_parameters.withdraw)
    self.sample_parameters_widgets = OrderedDict()

    parameters = ['Longueur', 'Largeur', 'Epaisseur']

    for i, parameter in enumerate(parameters):
      self.add_label(frame=self.sample_parameters,
                     widgets_dict=self.sample_parameters_widgets,
                     text=parameter,
                     name=parameter + '_label')
      self.sample_parameters_widgets[parameter + '_label'].grid(row=i, column=0)

      self.add_entry(frame=self.sample_parameters,
                     widgets_dict=self.sample_parameters_widgets,
                     entry_name=parameter)
      self.sample_parameters_widgets[parameter].grid(row=i, column=1)

    self.add_button(frame=self.sample_parameters,
                    widgets_dict=self.sample_parameters_widgets,
                    name="ok_button",
                    text="Fini",
                    command_type='custom',
                    command=self.sample_parameters.withdraw)
    self.sample_parameters_widgets["ok_button"].grid(row=3, column=0,
                                                     columnspan=2)


    # for i, parameter in enumerate(self.sample_parameters_widgets.values()[::2]):
    #   parameter.grid(row=i, column=0)
    # for i, parameter in enumerate(
    #     self.sample_parameters_widgets.values()[1::2]):
    #   parameter.grid(row=i, column=1)

  def create_popup_command(self):

    if hasattr(self, 'popup_command'):
      self.popup_command.deiconify()
      return
    self.check_goto_bool = False
    self.popup_command = tk.Toplevel()
    self.popup_command.title("Aller à")
    self.popup_command.protocol("WM_DELETE_WINDOW",
                                self.popup_command.withdraw)

    self.popup_command_widgets = OrderedDict()
    combobox_entries = ('Effort(N)', 'Position(mm)', 'Position(%)')

    self.add_label(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   name="command_type_label",
                   text="Type de commande")

    self.add_combobox(frame=self.popup_command,
                      widgets_dict=self.popup_command_widgets,
                      entries=combobox_entries,
                      name='command_type',
                      variable="command_type")

    self.add_label(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   name="command_value_label",
                   text="Valeur")

    self.add_entry(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   entry_name='command_value',
                   vartype=tk.DoubleVar())

    self.add_button(frame=self.popup_command,
                    widgets_dict=self.popup_command_widgets,
                    name="submit_command_button",
                    text="Go!",
                    command_type="custom",
                    command=lambda: self.init_command())

    toto = self.popup_command_widgets.values()
    del toto[1]

    for i, widg in enumerate(toto):
      widg.grid(row=0, column=i)

    self.add_button(frame=self.popup_command,
                    widgets_dict=self.popup_command_widgets,
                    name="quit_popup_command",
                    text='Fini',
                    command=self.popup_command.withdraw,
                    command_type="custom")

    self.popup_command_widgets["quit_popup_command"].grid(row=0, column=5)

  def add_scale(self, **kwargs):
    widgets_dict = kwargs.get('widgets_dict', None)
    frame = kwargs.get('frame', None)
    name = kwargs.get('name', 'Button')
    boundaries = kwargs.get("boundaries", (0, 1))

    widgets_dict[name] = tk.Scale(frame,
                                  from_=boundaries[0],
                                  to_=boundaries[1],
                                  orient=tk.HORIZONTAL,
                                  )

  def create_popup_speed(self):
    if hasattr(self, 'popup_speed'):
      self.popup_speed.deiconify()
      return

    self.popup_speed = tk.Toplevel()
    self.popup_speed.title("Vitesse du moteur")
    self.popup_speed.protocol("WM_DELETE_WINDOW",
                              self.popup_speed.withdraw)

    self.popup_speed_widgets = OrderedDict()

    self.add_label(frame=self.popup_speed,
                   widgets_dict=self.popup_speed_widgets,
                   text="Vitesse du moteur (mm/s)",
                   name="vit_mot_label")

    self.add_scale(frame=self.popup_speed,
                   widgets_dict=self.popup_speed_widgets,
                   name="vit_mot_scale",
                   boundaries=(0, 255))

    self.add_button(frame=self.popup_speed,
                    widgets_dict=self.popup_speed_widgets,
                    name="vit_mot_submit",
                    command="VITESSE",
                    text='Soumettre')

    for value in self.popup_speed_widgets.values():
      value.pack()

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
                     self.cycles_widgets["maximum_type"].get(),
                     self.maximum_entry.get(),
                     self.cycles_widgets["minimum_type"].get(),
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

  def init_command(self):
    try:
      self.command_value.get()
    except ValueError:
      return

    self.command_bool = True
    self.check_goto_bool = True

  def check_go_to(self, pos, eff, prct, sns):

    var_type = self.popup_command_widgets["command_type_combobox"].get()
    if var_type == "Effort(N)":
      var = eff
    elif var_type == "Position(mm)":
      var = pos
    elif var_type == "Position(%)":
      var = prct

    if var < self.command_value.get() and self.command_bool:
      self.submit_command("TRACTION")
      self.command_bool = False
    elif var > self.command_value.get() and self.command_bool:
      self.submit_command("COMPRESSION")
      self.command_bool = False

    if var >= self.command_value.get() and sns == 1:
      self.submit_command("STOP")
      self.check_goto_bool = False
    elif var <= self.command_value.get() and sns == -1:
      self.submit_command("STOP")
      self.check_goto_bool = False

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
    elif arg == "VITESSE":
      dico = {"vts": self.popup_speed_widgets["vit_mot_scale"].get()}
    else:
      dico = arg
    message = str(dico)
    self.queue.put(message)

  def update_data(self, message):

    try:
      eff = message.get('eff')
      sns = message.get('sns')
      mil = message.get('mil')
      pos = message.get('pos') # * 100 * 3 / 8668700.
    except (TypeError, AttributeError):
      pos = 0.0
      eff = 0.0
      sns = 0.0
      mil = 0.0

    prct = 0.0
    # if hasattr(self, 'popup_speed_widgets'):
    #   pos = message.get('pos') * self.popup_speed_widgets[
    #     "vit_mot_scale"].get() * (- 3 / (8668700. * 255)

    dico = {"position_abs": pos, "effort": eff, "sens": sns, "position_prct":
      prct}

    if hasattr(self, 'Longueur'):
      # longueur = self.sample_parameters_widgets['Longueur'].get()
      try:
        prct = (pos - self.Longueur.get()) / self.Longueur.get()
        dico["position_prct"] = prct
      except (ValueError, ZeroDivisionError):
        pass
    if hasattr(self, 'limits_widgets'):
      self.check_limits(eff, pos, sns, prct)
    if self.cycles_started:
      self.check_cycle(eff, sns, pos, prct)
    self.update_widgets(dico)
    if hasattr(self, 'popup_command'):
      if self.check_goto_bool:
        self.check_go_to(pos, eff, prct, sns)

  def update_widgets(self, message):

    mot = format(message["position_abs"], '.3f') + ' mm'
    self.displayer_widgets["position"].configure(text=mot)

    mot = format(message["effort"], '.2f') + ' N'
    self.displayer_widgets["effort"].configure(text=mot)

    if "position_prct" in message:
      mot = format(message["position_prct"], '.3f') + ' %'
      self.displayer_widgets["position_prct"].configure(text=mot)

    if message["sens"] == 1:
      self.action_widgets["TRACTION"].configure(bg="blue")
    elif message["sens"] == -1:
      self.action_widgets["COMPRESSION"].configure(bg="blue")
    else:
      self.action_widgets["COMPRESSION"].configure(bg="white")
      self.action_widgets["TRACTION"].configure(bg="white")

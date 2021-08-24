# coding: utf-8

from .frame_objects import FrameObjects
from collections import OrderedDict

from ..._global import OptionalModule

try:
  import tkinter as tk
  from tkinter import ttk
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")
  ttk = OptionalModule("tkinter")


class MinitensPopups(FrameObjects):
  """This class contains every popup. That means everything except the frames
  in the main window. The popups are called via a command in the menu.
  """

  def __init__(self):
    super().__init__()

  def create_popup_limits(self):

    if hasattr(self, 'popup_limits'):
      self.popup_limits.deiconify()
      return

    self.popup_limits = tk.Toplevel()
    self.popup_limits.resizable(False, False)
    self.popup_limits.protocol("WM_DELETE_WINDOW",
                               self.popup_limits.withdraw)

    self.popup_limits.title("Limites")
    self.create_menu_limits(self.popup_limits)

  def create_menubar(self):
    """A menubar that contains every menu."""

    # 1st menu: contains experience parameters.
    self.menubar = tk.Menu(self)
    self.menu_exp_parameters = tk.Menu(self.menubar, tearoff=0)
    self.menubar.add_cascade(label="Paramètres de l'essai",
                             menu=self.menu_exp_parameters)

    self.menu_exp_parameters.add_command(label='Limites',
                                         command=self.create_popup_limits)
    self.menu_exp_parameters.add_command(label='Vitesse',
                                         command=self.create_popup_speed)
    self.menu_exp_parameters.add_command(
      label='Paramètres échantillon',
      command=self.create_popup_sample_parameters)

    # 2nd menu: tools to conduct the experience.
    self.menu_tools = tk.Menu(self.menubar, tearoff=0)
    self.menubar.add_cascade(label="Outils",
                             menu=self.menu_tools)
    self.menu_tools.add_command(label='Consigne...',
                                command=self.create_popup_command)
    self.menu_tools.add_command(label="Générateur de cycles",
                                command=self.create_popup_new_cycle)
    self.menu_tools.add_command(label='Utilitaire de calibration',
                                command=self.create_popup_calibration)

    # 3rd menu: to enable or disable the link to crappy.
    self.menu_rec = tk.Menu(self.menubar, tearoff=0)
    self.menubar.add_cascade(label="Enregistrer?",
                             menu=self.menu_rec)

    self.menu_rec.add_checkbutton(label="Oui", variable=self.recording_state,
                                  onvalue=True, offvalue=False)

    self.menu_rec.add_checkbutton(label="Non", variable=self.recording_state,
                                  onvalue=False, offvalue=True)

    # 4th menu: to show on or show off some displays. Not working 20/07/2017
    # self.menu_display = tk.Menu(self.menubar, tearoff=0)
    # self.menubar.add_cascade(label="Affichage",
    #                          menu=self.menu_display)
    #
    # for key, value in self.variables.items():
    #   setattr(self, value + "_on", tk.BooleanVar())
    #   getattr(self, value + "_on").set(True)
    #   self.menu_display.add_checkbutton(label=key, variable=getattr(self,
    #                                                                 value +
    #                                                                 "_on"))

  def create_menu_limits(self, frame):
    """A menu to define limits of the minitens machine. It includes limits in
    every imaginable variable.
    """

    self.limits_widgets = OrderedDict()
    self.frame_limits = frame

    self.add_label(widgets_dict=self.limits_widgets,
                   frame=self.frame_limits,
                   text="Limites",
                   name="limites_title",
                   font=("Courier bold", 14, "bold"))
    self.limits_widgets['limites_title'].grid(row=0, columnspan=4)

    for i, (label, name) in enumerate([("MAXIMUM", "Haute"),
                                       ("MINIMUM", "Basse"),
                                       ("Actif?", "Actif")]):
      self.add_label(widgets_dict=self.limits_widgets,
                     frame=self.frame_limits,
                     text=label,
                     name=name)
      self.limits_widgets[name].grid(row=1, column=1 + i)

    for i, (variable, abrg) in enumerate(self.variables.items()):
      self.add_label(widgets_dict=self.limits_widgets,
                     frame=self.frame_limits,
                     text=variable,
                     name=abrg)
      self.limits_widgets[abrg].grid(row=2 + i, column=0, sticky=tk.W)

    for i, (variable, abrg) in enumerate(self.variables.items()):
      self.add_entry(widgets_dict=self.limits_widgets,
                     frame=self.frame_limits,
                     name=abrg + '_haute')
      self.add_entry(widgets_dict=self.limits_widgets,
                     frame=self.frame_limits,
                     name=abrg + '_basse')

      self.add_checkbutton(widgets_dict=self.limits_widgets,
                           frame=self.frame_limits,
                           name=abrg + '_chck')

      self.limits_widgets[abrg + '_haute'].grid(row=2 + i, column=1)
      self.limits_widgets[abrg + '_basse'].grid(row=2 + i, column=2)
      self.limits_widgets[abrg + '_chck'].grid(row=2 + i, column=3)

    self.add_button(widgets_dict=self.limits_widgets,
                    frame=self.frame_limits,
                    name="quit_window",
                    text="Fini",
                    command_type="custom",
                    command=self.popup_limits.withdraw)

    self.limits_widgets["quit_window"].grid(row=10, column=1, columnspan=2)

  def create_popup_new_cycle(self):
    """A popup used in the cycle generator."""

    if hasattr(self, 'cycles_popup'):
      # To make sure the popup is not already created.
      self.cycles_popup.deiconify()
      return

    self.cycles_popup = tk.Toplevel()
    self.cycles_popup.title("Nouveau cycle")
    self.cycles_popup.protocol("WM_DELETE_WINDOW",
                               self.cycles_popup.withdraw)
    self.cycles_popup.resizable(False, False)

    entries_combobox = self.variables.keys()

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

    for name in ['maximum', 'minimum']:
      self.add_entry(widgets_dict=self.cycles_widgets,
                     frame=self.cycles_popup,
                     name=name + "_entry")

      self.add_combobox(widgets_dict=self.cycles_widgets,
                        frame=self.cycles_popup,
                        name=name + "_type",
                        entries=entries_combobox,
                        default=1)

    self.add_entry(widgets_dict=self.cycles_widgets,
                   frame=self.cycles_popup,
                   name="nombre_entry",
                   vartype=tk.IntVar())

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.cycles_popup,
                    name="submit",
                    text="Ajouter",
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

    self.cycles_widgets["maximum_type"].grid(row=1, column=0)
    self.cycles_widgets["maximum_entry"].grid(row=1, column=1)
    self.cycles_widgets["minimum_type"].grid(row=1, column=2)
    self.cycles_widgets["minimum_entry"].grid(row=1, column=3)
    self.cycles_widgets["nombre_entry"].grid(row=1, column=4)
    self.cycles_widgets["submit"].grid(row=2, column=0, columnspan=2)
    self.cycles_widgets["quit"].grid(row=2, column=2, columnspan=2)

  def create_popup_sample_parameters(self):
    """To define sample parameters. Used to compute strengh(MPa)."""

    if hasattr(self, 'sample_parameters'):
      self.sample_parameters.deiconify()
      return

    self.sample_parameters = tk.Toplevel()
    self.sample_parameters.title("en mm")
    self.sample_parameters.resizable(False, False)
    self.sample_parameters.protocol("WM_DELETE_WINDOW",
                                    self.sample_parameters.withdraw)
    self.sample_parameters_widgets = OrderedDict()

    parameters = [('Largeur', self.ep_width),
                  ('Epaisseur', self.ep_depth)]

    for i, (parameter, variable) in enumerate(parameters):
      self.add_label(frame=self.sample_parameters,
                     widgets_dict=self.sample_parameters_widgets,
                     text=parameter,
                     name=parameter + '_label')

      self.sample_parameters_widgets[parameter + '_label'].grid(row=i,
                                                                column=0)

      self.add_entry(frame=self.sample_parameters,
                     widgets_dict=self.sample_parameters_widgets,
                     name=parameter,
                     variable=variable)
      self.sample_parameters_widgets[parameter].grid(row=i, column=1)

    self.add_button(frame=self.sample_parameters,
                    widgets_dict=self.sample_parameters_widgets,
                    name="ok_button",
                    text="Fini",
                    command_type='custom',
                    command=self.sample_parameters.withdraw)
    self.sample_parameters_widgets["ok_button"].grid(row=5, column=0,
                                                     columnspan=2)

  def create_popup_command(self):
    """Popup to show command options."""

    if hasattr(self, 'popup_command'):
      self.popup_command.deiconify()
      return

    self.check_goto_bool = False
    self.popup_command = tk.Toplevel()
    self.popup_command.title("Aller à")
    self.popup_command.protocol("WM_DELETE_WINDOW",
                                self.popup_command.withdraw)
    self.popup_command.resizable(False, False)

    self.popup_command_widgets = OrderedDict()

    combobox_entries = self.variables.keys()

    self.add_label(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   name="command_type_label",
                   text="Type de commande")

    self.add_combobox(frame=self.popup_command,
                      widgets_dict=self.popup_command_widgets,
                      entries=combobox_entries,
                      name='command_type')

    self.add_label(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   name="command_value_label",
                   text="Valeur")

    self.add_entry(frame=self.popup_command,
                   widgets_dict=self.popup_command_widgets,
                   name='command_value',
                   vartype=tk.DoubleVar())

    self.add_button(frame=self.popup_command,
                    widgets_dict=self.popup_command_widgets,
                    name="submit_command_button",
                    text="Go!",
                    command_type="custom",
                    command=lambda: self.init_command())

    self.add_button(frame=self.popup_command,
                    widgets_dict=self.popup_command_widgets,
                    name="quit_popup_command",
                    text='Fini',
                    command=self.popup_command.withdraw,
                    command_type="custom")

    for i, widg in enumerate(self.popup_command_widgets.values()):
      widg.grid(row=0, column=i)
    self.popup_command_widgets["quit_popup_command"].grid(row=0, column=6)

  def create_popup_speed(self):
    """To define motor speed. Only an entry."""

    if hasattr(self, 'popup_speed'):
      self.popup_speed.deiconify()
      return

    self.popup_speed = tk.Toplevel()
    self.popup_speed.title("Vitesse du moteur")
    self.popup_speed.protocol("WM_DELETE_WINDOW",
                              self.popup_speed.withdraw)

    self.popup_speed.resizable(False, False)

    self.popup_speed_widgets = OrderedDict()

    self.add_label(frame=self.popup_speed,
                   widgets_dict=self.popup_speed_widgets,
                   text="Vitesse du moteur \n"
                        "0.08 à 20.8 mm/min",
                   name="vit_mot_label")

    self.add_entry(frame=self.popup_speed,
                   widgets_dict=self.popup_speed_widgets,
                   name="vit_mot",
                   variable=self.speed)

    self.add_button(frame=self.popup_speed,
                    widgets_dict=self.popup_speed_widgets,
                    name="vit_mot_submit",
                    command="VITESSE",
                    text='Soumettre')

    self.add_button(frame=self.popup_speed,
                    widgets_dict=self.popup_speed_widgets,
                    name="vit_mot_quit",
                    text="Fini",
                    command_type="custom",
                    command=self.popup_speed.withdraw)

    for value in self.popup_speed_widgets.values():
      value.pack()

  def create_popup_length_init(self, **_):
    """The first popup when program is started. Used to define the distance
    between jaws (mm)."""

    self.popup_init = tk.Toplevel()
    self.popup_init.resizable(False, False)

    self.init_popup_widgets = OrderedDict()
    self.popup_init.title("Longueur entre les mors (mm)")
    self.add_label(widgets_dict=self.init_popup_widgets,
                   frame=self.popup_init,
                   text="Entrer la valeur initiale entre les mors ! (mm)",
                   font=("Courier bold", 11),
                   name="text_label")
    self.add_entry(widgets_dict=self.init_popup_widgets,
                   frame=self.popup_init,
                   name="length_init")

    self.add_button(widgets_dict=self.init_popup_widgets,
                    frame=self.popup_init,
                    command_type='custom',
                    name="quit",
                    command=lambda: self.popup_init.destroy(),
                    text='FINI')

    for widg in self.init_popup_widgets.itervalues():
      widg.pack()

  def create_popup_calibration(self):
    """Popup to recreate calibration of the machine."""

    if hasattr(self, 'popup_calibration'):
      return

    ok = tk.messagebox.askokcancel(title="Confirmation",
                                   message="Changer les paramètres de "
                                           "calibration?")
    if not ok:
      return
    else:
      self.popup_calibration_widgets = OrderedDict()
      self.popup_calibration = tk.Toplevel()
      self.popup_calibration.resizable(False, False)

      self.popup_calibration.title("Calibration de la cellule d'effort")

      self.add_label(widgets_dict=self.popup_calibration_widgets,
                     frame=self.popup_calibration,
                     text="Placer un effort connu,\n"
                          " puis entrer cette valeur (N):",
                     font=("Courier bold", 16))

      self.add_entry(widgets_dict=self.popup_calibration_widgets,
                     frame=self.popup_calibration,
                     name="calib")

      self.add_button(widgets_dict=self.popup_calibration_widgets,
                      frame=self.popup_calibration,
                      text="OK",
                      command="CALIBRATION")

      for widg in self.popup_calibration_widgets.values():
        widg.pack()


class MinitensFrames(FrameObjects):

  def __init__(self):
    super().__init__()

  def create_frame_display(self):
    """The frame that shows values of positions and efforts. Also includes tare
    buttons for force and relative positions."""

    for i, (variable, abrg) in enumerate(self.variables.items()):
      self.add_label(widgets_dict=self.displayer_widgets,
                     frame=self.frame_displayer,
                     text=variable,
                     name=abrg + '_label',
                     font=("Courier bold", 11, "bold"))

      self.add_label(widgets_dict=self.displayer_widgets,
                     frame=self.frame_displayer,
                     text='0.0',
                     font=("Courier bold", 48),
                     name=abrg + '_display')

      self.displayer_widgets[abrg + '_label'].grid(row=2 * i,
                                                   column=0,
                                                   columnspan=3,
                                                   sticky=tk.W)
      self.displayer_widgets[abrg + '_display'].grid(row=2 * i + 1,
                                                     column=1,
                                                     columnspan=2)

    self.add_button(widgets_dict=self.displayer_widgets,
                    frame=self.frame_displayer,
                    name="effort_tare",
                    text='Zéro',
                    command="effort_tare")

    self.add_button(widgets_dict=self.displayer_widgets,
                    frame=self.frame_displayer,
                    name="prct_tare",
                    text="Zéro",
                    command_type="custom",
                    command=self.tare)

    self.displayer_widgets["effort_tare"].grid(row=5, column=0, sticky=tk.W)
    self.displayer_widgets["prct_tare"].grid(row=3, column=0, sticky=tk.W)

  def create_frame_action(self):
    """The frame that shows the action buttons."""

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

    for ids, widget in enumerate(self.action_widgets):
      self.action_widgets[widget].grid(row=0 + 1, column=ids)

  def create_frame_cycles(self):

    self.add_label(widgets_dict=self.cycles_widgets,
                   frame=self.frame_cycles,
                   text="GENERATEUR DE CYCLES",
                   name="CYCLES",
                   font=("Courier bold", 14, "bold"))

    self.cycles_widgets["cycles_type"] = tk.StringVar()
    self.cycles_widgets["cycles_type"].set("position_prct")
    self.cycles = []
    self.nb_cycles = 0.
    self.cycles_started = False
    self.rising = True

    self.add_button(widgets_dict=self.cycles_widgets,
                    frame=self.frame_cycles,
                    name="start_cycle",
                    text="Démarrer",
                    bg="green",
                    command_type="custom",
                    command=self.start_cycle,
                    width=15, height=3)

    self.add_checkbutton(widgets_dict=self.cycles_widgets,
                         frame=self.frame_cycles,
                         variable="start_rec_cycle",
                         text="Au démarrage: lancer l'enregistrement",
                         name="start_rec_cycle")

    self.add_checkbutton(widgets_dict=self.cycles_widgets,
                         frame=self.frame_cycles,
                         variable="stop_rec_cycle",
                         text="A l'arrêt: stopper l'enregistrement",
                         name="stop_rec_cycle")

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
    self.cycles_widgets["start_cycle"].grid(row=1, column=3,
                                            columnspan=2, rowspan=3)
    self.cycles_widgets["start_rec_cycle"].grid(row=2, column=2, sticky=tk.W)
    self.cycles_widgets["stop_rec_cycle"].grid(row=3, column=2, sticky=tk.W)
    self.cycles_table.grid(row=5, column=0, columnspan=4, rowspan=1)


class MinitensFrame(MinitensFrames, MinitensPopups):
  def __init__(self, parent, **kwargs):
    """Frame to use with minitens in crappy."""

    tk.Frame.__init__(self, parent)
    # self.grid()

    self.frames = ["displayer",
                   "action",
                   "cycles"]

    for frame in self.frames:
      setattr(self, "frame_" + frame, tk.Frame(self,
                                               relief=tk.SUNKEN,
                                               borderwidth=1))
      setattr(self, frame + "_widgets", OrderedDict())

      self.variables = OrderedDict([("Position(mm)", "position"),
                                    ("Position(%)", "position_prct"),
                                    ("Effort(N)", "effort"),
                                    ("Contrainte(MPa)", "contrainte")])

    self.queue = kwargs.get("queue")
    self.crappy_queue = kwargs.get("crappy_queue")
    self.recording_state = tk.BooleanVar()
    self.create_widgets(**kwargs)

    for name, default in [('speed', 5),
                          ('ep_length', 0),
                          ('ep_width', 0),
                          ('ep_depth', 0),
                          ('length_0', 10)]:
      setattr(self, name, tk.DoubleVar())
      getattr(self, name).set(default)
    self.submit_command("VITESSE")

    # self.position = self.length_0.get()
    self.position = 0.0
    self.position_rel = 0.0
    self.position_prct = 0.0

    self.create_popup_length_init()
    self.wait_window(self.popup_init)

    try:
      self.position = self.length_init_var.get()
    except ValueError:
      self.position = 0
    self.position_prct = self.position

  def create_widgets(self, **_):
    """ Frames organization

      Frame displayer: to display effort and displacement values.
      Frame position: to start and stop the motor, according to pre-defined
      values.
      The cycle frame is also showed if cycles are defined.
    """

    self.create_menubar()
    self.create_frame_display()
    self.create_frame_action()
    self.create_frame_cycles()

    self.frame_displayer.grid(row=0, column=0)
    self.frame_action.grid(row=1, column=0)
    self.frame_cycles.grid(row=0, column=1)
    self.frame_cycles.grid_forget()

  def check_limits(self, **kwargs):
    """A method that checks every variable, and if a limit has been reached. In
    that case, the stop command is sent to arduino."""

    sens = kwargs.pop("sens")

    for variable, abrg in self.variables.items():
      try:
        if getattr(self, abrg + '_chck_var').get():
          var = kwargs.pop(abrg)
          if var <= getattr(self, abrg + '_basse_var').get() and sens == -1:
            self.submit_command("STOP")
          if var >= getattr(self, abrg + '_haute_var').get() and sens == 1:
            self.submit_command("STOP")
      except ValueError:
        pass

  def submit_cycle(self):
    """To submit new cycles. If the user entered correct parameters, will be
    added to the cycle generator."""

    try:
      nb_cycles = self.nombre_entry_var.get()

      for i in range(nb_cycles):
        new_cycle = [len(self.cycles) + 1,
                     self.maximum_type_var.get(),
                     self.maximum_entry_var.get(),
                     self.minimum_type_var.get(),
                     self.minimum_entry_var.get()]
        self.cycles.append(new_cycle)
        self.cycles_table.insert("", "end",
                                 text=new_cycle[0],
                                 values=new_cycle[1:])

    except ValueError:
      pass
    for widg in ['nombre_entry_var', 'minimum_entry_var', 'nombre_entry_var']:
      getattr(self, widg).set(0.0)

  def init_command(self):
    """Used if the user entered a command. Does nothing if incorrect type is
    entered."""

    try:
      self.command_value_var.get()
    except ValueError:
      return
    self.command_bool = True
    self.check_goto_bool = True

  def check_go_to(self, **kwargs):
    """Used if the user entered a command. Checks if the destination has been
    reached, and sends stop if it's the case."""

    sens = kwargs.pop("sens")
    var_type = self.command_type_var.get()

    for name, variable in self.variables.items():
      if name == var_type:
        var = kwargs.pop(variable)

    if self.command_bool:
      if var < self.command_value_var.get():
        self.submit_command("TRACTION")
        self.command_bool = False
      elif var > self.command_value_var.get():
        self.submit_command("COMPRESSION")
        self.command_bool = False

    else:
      if var >= self.command_value_var.get() and sens == 1:
        self.submit_command("STOP")
        self.check_goto_bool = False

      elif var <= self.command_value_var.get() and sens == -1:
        self.submit_command("STOP")
        self.check_goto_bool = False

  def start_cycle(self):
    """Executed when cycle generator is started."""

    if self.cycles:
      self.cycles_started = True
      # var_type = self.cycles[0][1]

      self.submit_command("TRACTION")
      if self.start_rec_cycle.get():
        self.recording_state.set(True)

  def check_cycle(self, **kwargs):
    """Executed in case the cycle generator is started."""

    sens = kwargs.pop("sens")
    mode_max, maximum, mode_min, minimum = self.cycles[0][1:]

    for name, variable in self.variables.items():
      if name == mode_max:
        mode_max = variable
      if name == mode_min:
        mode_min = variable

    if self.rising:
      var = kwargs.pop(mode_max)
    else:
      var = kwargs.pop(mode_min)

    if var >= maximum and sens == 1 and self.rising:
      self.nb_cycles += 0.5
      self.rising = False
      self.submit_command("COMPRESSION")

    elif var <= minimum and sens == -1 and not self.rising:
      self.nb_cycles += 0.5
      self.rising = True

      del self.cycles[0]
      self.cycles_table.delete(self.cycles_table.get_children()[0])

      if self.cycles:
        self.submit_command("TRACTION")
      else:
        self.cycles_started = False
        self.submit_command("STOP")
        if self.stop_rec_cycle.get():
          self.recording_state.set(False)

  def tare(self):
    """Tare for position_prct variable."""

    self.position_prct = self.position

  def calc_speed(self):
    """Does the conversion between speed and bytes, to send to arduino. Also
    makes sure the user entered in the right range."""

    if self.speed.get() < 0.08:
      self.speed.set(0.08)
    elif self.speed.get() > 20.8:
      self.speed.set(20.8)
    return int(255 * (-self.speed.get() / 20.8 + 1))

  def submit_command(self, arg):
    """Information to transmit to the arduino."""

    if arg == "STOP":
      dico = {"s": 0}
    elif arg == "TRACTION":
      dico = {"s": 1}
    elif arg == "COMPRESSION":
      dico = {"s": -1}
    elif arg == "effort_tare":
      dico = {"t": 0}
    elif arg == "VITESSE":
      dico = {"v": self.calc_speed()}
    elif arg == "CALIBRATION":
      self.popup_calibration.destroy()
      del self.popup_calibration
      dico = {"c": self.calib_var.get()}
    else:
      dico = arg
    message = str(dico)
    self.queue.put(message)

  def update_data(self, new_data):
    """Retrieves data from the ArduinoHandler, updates the GUI, and sends it to
    the crappy link."""

    # Retrieved from ArduinoHandler
    effort = new_data.get('e')
    sens = new_data.get('s')
    millis = new_data.get('m')
    position_abs = new_data.get('p')

    # Computed
    delta = position_abs - self.position_rel
    self.position_rel = position_abs
    self.position += delta * self.speed.get() / (1000 * 60.)

    try:
      # The try..except to prevent user to enter wrong data type for sample
      # parameters.
      if 0 in (self.ep_depth.get(), self.ep_width.get()):
        # In case the user didn't entered sample parameters.
        contrainte = 0.0
      else:
        contrainte = effort / (self.ep_depth.get() * self.ep_width.get())
    except ValueError:
      contrainte = 0.0

    if not self.position_prct == 0.:
      # In case the user didn't initialized distance between claws.
      position_prct = 100 * (self.position - self.position_prct) / \
        self.position_prct
    else:
      position_prct = 0.0

    if hasattr(self, 'limits_widgets'):
      self.check_limits(effort=effort,
                        position=self.position,
                        sens=sens,
                        position_prct=position_prct,
                        contrainte=contrainte)

    if self.cycles_started:
      self.check_cycle(effort=effort,
                       sens=sens,
                       position=self.position,
                       position_prct=position_prct,
                       contrainte=contrainte)

    if hasattr(self, 'popup_command'):

      if self.check_goto_bool:
        self.check_go_to(position=self.position,
                         position_prct=position_prct,
                         contrainte=contrainte,
                         effort=effort,
                         sens=sens)

    to_send = {"position": self.position,
               "effort": effort,
               "sens": sens,
               "position_prct": position_prct,
               "contrainte": contrainte,
               "millis": millis}

    self.update_widgets(to_send)

    if self.recording_state.get():
      # for key, value in to_send.items():
      #   to_send[self.variables[key]] =
      self.crappy_queue.put(to_send)

  def update_widgets(self, new_data):
    formating_parameters = [("position", ('.3f', 'mm')),
                            ("effort", ('.2f', 'N')),
                            ('position_prct', ('.2f', '%')),
                            ('contrainte', ('.3f', 'MPa'))]

    for parameters, formats in formating_parameters:
      mot = format(new_data[parameters], formats[0]) + ' ' + formats[1]
      self.displayer_widgets[parameters + "_display"].configure(text=mot)
    if new_data["sens"] == 1:
      self.action_widgets["TRACTION"].configure(bg="blue")
      self.action_widgets["COMPRESSION"].configure(bg="white")
    elif new_data["sens"] == -1:
      self.action_widgets["COMPRESSION"].configure(bg="blue")
      self.action_widgets["TRACTION"].configure(bg="white")
    else:
      self.action_widgets["COMPRESSION"].configure(bg="white")
      self.action_widgets["TRACTION"].configure(bg="white")

    if self.cycles:
      self.frame_cycles.grid()
    else:
      self.frame_cycles.grid_forget()

      # for key, value in self.variables.items():
      #   if not getattr(self, value + '_on').get():
      #     self.displayer_widgets[value + "_label"].grid_forget()
      #     self.displayer_widgets[value + "_display"].grid_forget()
      #   else:
      #     self.displayer_widgets[value + "_label"].grid()
      #     self.displayer_widgets[value + "_display"].grid()

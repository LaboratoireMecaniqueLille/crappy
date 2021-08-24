# coding:utf-8

import Tix
from Tkinter import *
from crappy.actuator import actuator_list


class Interface(Frame):
  """Creates a graphic interface that allows to connect the actuator via a
  serial connection and to send command in terms of speed or position."""

  def __init__(self, root, **kwargs):
    Frame.__init__(self, root, width=1000, height=1000, **kwargs)
    root.geometry("395x320")
    root.title("Pilotage")

    # Création des onglets
    monnotebook = Tix.NoteBook(root)
    monnotebook.add("page1", label="Configuration")
    monnotebook.add("page2", label="Velocity Mode")
    monnotebook.add("page3", label="Motion Task")

    p1 = monnotebook.subwidget_list["page1"]
    p2 = monnotebook.subwidget_list["page2"]
    p3 = monnotebook.subwidget_list["page3"]

    fra1 = Canvas(p1)
    fra2 = Canvas(p2)
    fra3 = Canvas(p3)

    fra1.pack(expand=1, fill=Tix.BOTH)
    fra2.pack(expand=1, fill=Tix.BOTH)
    fra3.pack(expand=1, fill=Tix.BOTH)
    # fin création des onglets

    # Début onglet 1
    # create a frame in canvas(p1)
    sous_fra = Frame(fra1, width=400, borderwidth=2, relief=GROOVE)
    self.portLabel = Label(sous_fra, text="Serial ser:")  # create a label
    # create a variable, it will contain ser selection
    # self.myPortCombo = Tix.StringVar()
    self.myPortCombo = Tix.StringVar(sous_fra, "/dev/ttyUSB0")

    self.actuatorNameCombo = Tix.StringVar()
    self.actuatorNameCombo.set("CM_drive")
    # create a combobox, it will contain names of ports
    self.actuatorCombo = Tix.ComboBox(sous_fra, editable=1, dropdown=1,
                                      variable=self.actuatorNameCombo)
    # configure the combobox in read only
    self.actuatorCombo.entry.config(state='readonly')
    for m in actuator_list:
      self.actuatorCombo.insert(0, m)

    self.actuatorLabel = Label(sous_fra, text="Actuator name:")
    self.portCombo = Entry(sous_fra, textvariable=self.myPortCombo)
    self.baudrateLabel = Label(sous_fra, text="Baudrate:")  # create a label
    # create a variable, it will contain baudrate selection
    # self.baudCombo = Tix.StringVar()
    self.baudCombo = Tix.StringVar(sous_fra, "9600")
    self.baudrateCombo = Entry(sous_fra, textvariable=self.baudCombo)
    self.num_device = Tix.IntVar()
    # create a label
    self.num_device_label = Label(sous_fra, text="Device number:")
    self.num_device_entry = Spinbox(sous_fra, from_=1, to=10,
                                    textvariable=self.num_device)
    self.actuatorNameCombo.trace('w', self.callback)
    # create connect Button
    self.connection = Button(sous_fra, text="Connect", command=self.connection)
    sous_fra1 = Frame(fra1, width=400)  # create a second frame in canvas(p1)
    # create examine location button
    self.location = Button(sous_fra1, text="Examine location",
                           command=self.examine_location, width=10)
    self.location.config(state='disabled')  # disable the state of the button
    # create 'location:' label
    location_label = Label(sous_fra1, text="Location:")
    # create a variable, it will contain the result of examine_location method
    self.Resultat = StringVar()
    self.Resultat.set("0")  # set the variable to zero
    # create a label, it will show the variable
    resultat_label = Label(sous_fra1, textvariable=self.Resultat)
    self.resetZero = Button(sous_fra1, text=" Reset ",
                            command=self.reset_servostar)
    # create cancel button
    self.quit = Button(sous_fra1, text="Cancel", command=root.quit)
    # Fin onglet 1

    # Début onglet 2
    # create a frame in canvas(p2)
    sous_fra2 = Frame(fra2, width=400)
    # create a second frame in canvas(p2)
    sous_fra2bis = Frame(fra2, width=400)
    # create a variable, it will contain the value of speed_entry
    self.speedVar = StringVar()
    # create 'Velocity:' label
    speed_label = Label(sous_fra2bis, text="Velocity:")
    # create an entry, it will contain a velocity choose by the user
    speed_entry = Entry(sous_fra2bis, textvariable=self.speedVar, width=5)
    # create a advance button
    self.advanceButton = Button(sous_fra2bis, text="advance",
                                command=self.advance, width=10)
    # disable the state of the button
    self.advanceButton.config(state='disabled')
    # create recoil button
    self.recoilButton = Button(sous_fra2bis, text="recoil",
                               command=self.recoil, width=10)
    # disable the state of the button
    self.recoilButton.config(state='disabled')
    # create stop button
    self.stopVMButton = Button(sous_fra2bis, text="STOP",
                               command=self.stop_motion, width=10)
    self.stopVMButton.config(state='disabled')
    self.defineZeroButton = Button(sous_fra2bis, text="Define Zero",
                                   command=self.define_zero, width=10)
    self.defineZeroButton.config(state='disabled')
    # Début onglet 3
    sous_fra3_2 = Frame(fra3, width=400)  # create a frame in canvas(p3)
    self.moveZeroButton = Button(sous_fra3_2, text="Move Home",
                                 command=self.move_zero, width=15)
    # create move home button

    # disable the state of the button
    self.moveZeroButton.config(state='disabled')

    # create 'position:' label
    position_label = Label(sous_fra3_2, text="position :")
    # create a variable, it will contain the value of entry2
    self.position = StringVar()
    # create an entry, it will contain the positon choose by the user
    self.entry2 = Entry(sous_fra3_2, textvariable=self.position)
    # pick out the widget that will receive keyboard events
    self.entry2.focus_set()

    speed_label = Label(sous_fra3_2, text="speed:")  # create 'position:' label
    # create a variable, it will contain the value of entry2
    self.speed = StringVar()
    # create an entry, it will contain the positon choose by the user
    self.entry3 = Entry(sous_fra3_2, textvariable=self.speed)
    # pick out the widget that will receive keyboard events
    self.entry3.focus_set()

    # create 'motionType:' label
    motion_type_label = Label(sous_fra3_2, text="motion type :")
    # create a variable, it will contain the value of entry4
    self.motionType = StringVar()

    sous_fra3 = Frame(fra3, width=400)  # create a frame in canvas(p3)
    sous_sous_fra3 = Frame(sous_fra3, width=400)  # create a frame in sous_fra3
    # create move button
    self.moveButton = Button(sous_sous_fra3, text="Move", command=self.move)
    self.moveButton.config(state='disabled')  # disable the state of the button
    # create STOP button
    self.stopMTButton = Button(sous_sous_fra3, text="STOP",
                               command=self.stop_motion)
    self.stopMTButton.config(state='disabled')
    # create a second frame in sous_fra3
    sous_sous_fra3bis = Frame(sous_fra3, width=400)
    # placement des widgets onglet 1
    # show widgets on canvas(p1)
    sous_fra.grid(padx=10, pady=10)  #
    self.portLabel.grid(row=1, column=0, sticky="sw", padx=10, pady=10)
    self.portCombo.grid(row=1, column=1, sticky="sw", padx=10, pady=10)
    self.baudrateLabel.grid(row=2, column=0, sticky="sw", padx=10, pady=10)
    self.baudrateCombo.grid(row=2, column=1, sticky="sw", padx=10, pady=10)
    self.actuatorLabel.grid(row=3, column=0, sticky="sw", padx=10, pady=10)
    self.actuatorCombo.grid(row=3, column=1, sticky="sw", padx=10, pady=10)
    self.connection.grid(row=5, column=1, sticky="se")
    # placement widget sous frame onglet 1
    # show widgets on frame canvas(p1)
    sous_fra1.grid()
    # self.enable.grid(row=1, column=0,sticky= "sw", padx=10,pady=10)
    # self.disable.grid(row=2, column=0,sticky= "sw", padx=10,pady=10)
    # self.resetZero.grid(row=3, column=0,sticky= "sw", padx=10,pady=10)
    self.location.grid(row=5, column=0, sticky="s", padx=10, pady=10)
    location_label.grid(row=5, column=1)
    resultat_label.grid(row=5, column=2)
    self.resetZero.grid(row=5, column=4)
    # self.quit.grid(row=4, column=4, sticky= "e")

    # placement des widgets onglet 2
    # show widgets on canvas(p2)
    sous_fra2.grid(row=0, column=0, padx=10, pady=10)
    sous_fra2bis.grid(row=1, column=0, padx=10, pady=10)
    # self.init_modeButton.grid(row=0, column=0, padx=10,pady=10)
    speed_label.grid(row=0, column=0, sticky='w')
    speed_entry.grid(row=0, column=2, sticky='w')
    self.recoilButton.grid(row=0, column=3, sticky='w')
    self.advanceButton.grid(row=1, column=3, sticky='w')
    self.stopVMButton.grid(row=2, column=3, sticky='w')
    self.defineZeroButton.grid(row=3, column=3, sticky='w')

    # placement des widgets onglet 3
    #  show widgets on canvas(p3)
    sous_fra3_2.grid(padx=10, pady=10)
    self.moveZeroButton.grid(row=0, column=0, padx=10, pady=10)
    position_label.grid(row=1, column=0, sticky='w')
    speed_label.grid(row=2, column=0, sticky='w')
    motion_type_label.grid(row=3, column=0, sticky='w')
    self.entry2.grid(row=1, column=1, sticky='w')
    self.entry3.grid(row=2, column=1, sticky='w')
    Radiobutton(sous_fra3_2, text="absolute", variable=self.motionType,
                value=True).grid()
    Radiobutton(sous_fra3_2, text="relative", variable=self.motionType,
                value=False).grid()
    sous_fra3.grid(row=3, column=0)
    sous_sous_fra3.grid(row=0, column=0)
    sous_sous_fra3bis.grid(row=1, column=0)
    self.moveButton.grid(row=0, column=1)
    self.stopMTButton.grid(row=0, column=2)
    # show notebooks
    monnotebook.pack(side=LEFT, fill=Tix.BOTH, expand=1, padx=5, pady=5)

  # function to initialize the connection
  def connection(self):
    if "" in [self.myPortCombo.get(), self.baudCombo.get(),
              self.actuatorNameCombo.get()]:
      print('you must choose actuator name and configuration.')
    else:
      try:
        actuator = actuator_list[self.actuatorNameCombo.get()]
        if self.actuatorNameCombo.get().capitalize() == "Oriental":
          self.actuator = actuator(port=self.myPortCombo.get(),
                                   num_device=int(self.num_device_entry.get()),
                                   baudrate=self.baudCombo.get())
        else:
          self.actuator = actuator(port=self.myPortCombo.get(),
                                   baudrate=self.baudCombo.get())
        self.actuatorName = self.actuatorNameCombo.get()
        print('connection')
        self.actuator.open()
        self.location.config(state='normal')
        self.moveButton.config(state='normal')
        self.recoilButton.config(state='normal')
        self.advanceButton.config(state='normal')
        self.moveZeroButton.config(state='normal')
        self.defineZeroButton.config(state='normal')
        self.stopMTButton.config(state='normal')
        self.stopVMButton.config(state='normal')

      except Exception as e:
        print(' Connection error:', e)

  def callback(self, *_):
    # print args
    # print 'test: ', self.actuatorNameCombo.get().capitalize()
    if self.actuatorNameCombo.get().capitalize() == "Oriental":
      self.num_device_label.grid(row=4, column=0, sticky="sw", padx=10,
                                 pady=10)
      self.num_device_entry.grid(row=4, column=1, sticky="sw", padx=10,
                                 pady=10)
    else:
      self.num_device_entry.grid_forget()
      self.num_device_label.grid_forget()

  def define_zero(self):
    if self.actuatorName == "CmDrive":
      self.actuator.ser.close()
      self.actuator.ser.open()
      self.actuator.ser.write('P=0\r')
      # self.actuator.ser.readline()
      self.actuator.ser.close()
    else:
      try:
        self.actuator.set_home()
      except NotImplementedError:
        print("Not implemented yet.")

  def reset_servostar(self):
    self.actuator.reset()

  # function to examine the location of the actuator
  def examine_location(self):
    location = self.actuator.get_position()
    self.Resultat.set(location)

  # function to move home
  def move_zero(self):
    self.actuator.move_home()
    print('moving home')

  # function to apply a motion task
  def move(self):
    if self.motionType.get() == "" or self.position.get() == "":
      print("one of the entry is empty")
    else:
      if self.actuatorName == "CmDrive":
        if self.motionType.get() == '1':
          self.actuator.set_position(int(self.position.get()), None,
                                     'absolute')
          print('MA mode')
        else:
          self.actuator.set_position(int(self.position.get()), None,
                                     'relative')
          print('MR mode')
      else:
        self.actuator.set_position(int(self.position.get()),
                                   int(self.speed.get()))

  # function to advance the actuator on velocity mode
  def advance(self):
    if self.speedVar.get() == "":
      print('choose velocity')
    else:
      self.actuator.set_speed(int(self.speedVar.get()))
      print('the actuator goes up with speed=%i' % int(self.speedVar.get()))

  # function to recoil the actuator on velocity mode
  def recoil(self):
    if self.speedVar.get() == "":
      print('choose velocity')
    else:
      self.actuator.set_speed(-int(self.speedVar.get()))
      print('the actuator goes down with speed=%i' % int(self.speedVar.get()))

  # function to stop a motion
  def stop_motion(self):
    self.actuator.stop()
    print('Motion has been stopped')


if __name__ == '__main__':
  root = Tix.Tk()
  interface = Interface(root)
  interface.mainloop()
  interface.destroy()

# -*- coding:utf-8 -*-
import Tix
from Tkinter import *
import Tkinter
# from serial.tools import list_ports
from crappy2.technical import __motors__ as motors


class Interface(Frame):
    """
    Creat a graphic interface that permit to connect the motor via a serial ser,
    and to send command in terms of speed or position
    """

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
        sous_fra = Frame(fra1, width=400, borderwidth=2, relief=GROOVE)  # create a frame in canvas(p1)
        self.portLabel = Label(sous_fra, text="Serial ser:")  # create a label
        self.myPortCombo = Tix.StringVar()  # create a variable, it will contain ser selection

        self.motorNameCombo = Tix.StringVar()
        self.motorCombo = Tix.ComboBox(sous_fra, editable=1, dropdown=1,
                                       variable=self.motorNameCombo)  # create a combobox, it will contain names of ports
        self.motorCombo.entry.config(state='readonly')  # configure the combobox in read only
        for i in range(len(motors)):
            self.motorCombo.insert(0, motors[i])

        self.motorLabel = Label(sous_fra, text="Motor name:")
        self.portCombo = Entry(sous_fra, textvariable=self.myPortCombo)
        self.baudrateLabel = Label(sous_fra, text="Baudrate:")  # create a label
        self.baudCombo = Tix.StringVar()  # create a variable, it will contain baudrate selection
        self.baudrateCombo = Entry(sous_fra, textvariable=self.baudCombo)
        self.num_device = Tix.IntVar()
        self.num_device_label = Label(sous_fra, text="Device number:")  # create a label
        self.num_device_entry = Spinbox(sous_fra, from_=1, to=10, textvariable=self.num_device)
        self.motorNameCombo.trace('w', self.callback)
        self.connection = Button(sous_fra, text="Connect", command=self.connection)  # create connect Button
        sous_fra1 = Frame(fra1, width=400)  # create a second frame in canvas(p1)
        self.location = Button(sous_fra1, text="Examine location", command=self.examineLocation,
                               width=10)  # create examine location button
        self.location.config(state='disabled')  # disable the state of the button
        locationLabel = Label(sous_fra1, text="Location:")  # create 'location:' label
        self.Resultat = StringVar()  # create a variable, it will contain the result of examineLocation method
        self.Resultat.set("0")  # set the variable to zero
        resultatLabel = Label(sous_fra1, textvariable=self.Resultat)  # create a label, it will show the variable
        self.resetZero = Button(sous_fra1, text=" Reset ", command=self.reset_servostar)
        self.quit = Button(sous_fra1, text="Cancel", command=root.quit)  # create cancel button
        # Fin onglet 1

        # Début onglet 2
        sous_fra2 = Frame(fra2, width=400)  # create a frame in canvas(p2)
        sous_fra2bis = Frame(fra2, width=400)  # create a second fram in canvas(p2)
        self.speedVar = StringVar()  # creaate a variable, it will contain the value of speedEntry
        speedLabel = Label(sous_fra2bis, text="Velocity:")  # create 'Velocity:' label
        speedEntry = Entry(sous_fra2bis, textvariable=self.speedVar,
                           width=5)  # create an entry, it will contain a velocity choose by the user
        self.advanceButton = Button(sous_fra2bis, text="advance", command=self.advance,
                                    width=10)  # create a advance button
        self.advanceButton.config(state='disabled')  # disable the state of the button
        self.recoilButton = Button(sous_fra2bis, text="recoil", command=self.recoil, width=10)  # create recoil button
        self.recoilButton.config(state='disabled')  # disable the state of the button
        self.stopVMButton = Button(sous_fra2bis, text="STOP", command=self.stopMotion, width=10)  # create stop button
        self.stopVMButton.config(state='disabled')
        self.defineZeroButton = Button(sous_fra2bis, text="Define Zero", command=self.defineZero, width=10)
        self.defineZeroButton.config(state='disabled')
        # Début onglet 3
        sous_fra3_2 = Frame(fra3, width=400)  # create a frame in canvas(p3)
        self.moveZeroButton = Button(sous_fra3_2, text="Move Home", command=self.moveZero,
                                     width=15)  # create move home button
        self.moveZeroButton.config(state='disabled')  # disable the state of the button

        positionLabel = Label(sous_fra3_2, text="position :")  # create 'position:' label
        self.position = StringVar()  # create a variable, it will contain the value of entry2
        self.entry2 = Entry(sous_fra3_2,
                            textvariable=self.position)  # create an entry, it will contain the positon choose by the user
        self.entry2.focus_set()  # pick out the widget that will receive keyboard events

        speed_label = Label(sous_fra3_2, text="speed:")  # create 'position:' label
        self.speed = StringVar()  # create a variable, it will contain the value of entry2
        self.entry3 = Entry(sous_fra3_2,
                            textvariable=self.speed)  # create an entry, it will contain the positon choose by the user
        self.entry3.focus_set()  # pick out the widget that will receive keyboard events

        motionTypeLabel = Label(sous_fra3_2, text="motion type :")  # create 'motionType:' label
        self.motionType = StringVar()  # create a variable, it will contain the value of entry4

        sous_fra3 = Frame(fra3, width=400)  # create a frame in canvas(p3)
        sous_sous_fra3 = Frame(sous_fra3, width=400)  # create a frame in sous_fra3
        self.moveButton = Button(sous_sous_fra3, text="Move", command=self.move)  # create move button
        self.moveButton.config(state='disabled')  # disable the state of the button
        self.stopMTButton = Button(sous_sous_fra3, text="STOP", command=self.stopMotion)  # create STOP button
        self.stopMTButton.config(state='disabled')
        sous_sous_fra3bis = Frame(sous_fra3, width=400)  # create a second frame in sous_fra3
        # placement des widgets onglet 1
        # show widgets on canvas(p1)
        sous_fra.grid(padx=10, pady=10)  #
        self.portLabel.grid(row=1, column=0, sticky="sw", padx=10, pady=10)
        self.portCombo.grid(row=1, column=1, sticky="sw", padx=10, pady=10)
        self.baudrateLabel.grid(row=2, column=0, sticky="sw", padx=10, pady=10)
        self.baudrateCombo.grid(row=2, column=1, sticky="sw", padx=10, pady=10)
        # self.num_device_label.grid(row=4, column=0, sticky="sw", padx=10, pady=10)
        # self.num_device_entry.grid(row=4, column=1, width=2, sticky="sw", padx=10, pady=10)
        self.motorLabel.grid(row=3, column=0, sticky="sw", padx=10, pady=10)
        self.motorCombo.grid(row=3, column=1, sticky="sw", padx=10, pady=10)
        self.connection.grid(row=5, column=1, sticky="se")
        # placement widget sous frame onglet 1
        # show widgets on frame canvas(p1)
        sous_fra1.grid()
        # self.enable.grid(row=1, column=0,sticky= "sw", padx=10,pady=10)
        # self.disable.grid(row=2, column=0,sticky= "sw", padx=10,pady=10)
        # self.resetZero.grid(row=3, column=0,sticky= "sw", padx=10,pady=10)
        self.location.grid(row=5, column=0, sticky="s", padx=10, pady=10)
        locationLabel.grid(row=5, column=1)
        resultatLabel.grid(row=5, column=2)
        self.resetZero.grid(row=5, column=4)
        # self.quit.grid(row=4, column=4, sticky= "e")

        # placement des widgets onglet 2
        # show widgets on canvas(p2)
        sous_fra2.grid(row=0, column=0, padx=10, pady=10)
        sous_fra2bis.grid(row=1, column=0, padx=10, pady=10)
        # self.init_modeButton.grid(row=0, column=0, padx=10,pady=10)
        speedLabel.grid(row=0, column=0, sticky='w')
        speedEntry.grid(row=0, column=2, sticky='w')
        self.recoilButton.grid(row=0, column=3, sticky='w')
        self.advanceButton.grid(row=1, column=3, sticky='w')
        self.stopVMButton.grid(row=2, column=3, sticky='w')
        self.defineZeroButton.grid(row=3, column=3, sticky='w')

        # placement des widgets onglet 3
        #  show widgets on canvas(p3)
        sous_fra3_2.grid(padx=10, pady=10)
        self.moveZeroButton.grid(row=0, column=0, padx=10, pady=10)
        positionLabel.grid(row=1, column=0, sticky='w')
        speed_label.grid(row=2, column=0, sticky='w')
        motionTypeLabel.grid(row=3, column=0, sticky='w')
        self.entry2.grid(row=1, column=1, sticky='w')
        self.entry3.grid(row=2, column=1, sticky='w')
        Radiobutton(sous_fra3_2, text="absolute", variable=self.motionType, value=True).grid()
        Radiobutton(sous_fra3_2, text="relative", variable=self.motionType, value=False).grid()
        sous_fra3.grid(row=3, column=0)
        sous_sous_fra3.grid(row=0, column=0)
        sous_sous_fra3bis.grid(row=1, column=0)
        self.moveButton.grid(row=0, column=1)
        self.stopMTButton.grid(row=0, column=2)
        # show notebooks
        monnotebook.pack(side=LEFT, fill=Tix.BOTH, expand=1, padx=5, pady=5)

    # function to initialize the connection 	    
    def connection(self):
        if self.myPortCombo.get() == "" or self.baudCombo.get() == "" or self.motorNameCombo.get() == "":
            print 'you must choose motor name and configuration.'
        else:
            try:
                try:
                    module = __import__("crappy2.technical", fromlist=[self.motorNameCombo.get()])
                    Motor = getattr(module, self.motorNameCombo.get())
                except Exception as e:
                    print "{0}".format(e), " : Unreconized motor\n"
                    return

                if self.motorNameCombo.get().capitalize() == "Oriental":
                    self.motor = Motor(port=self.myPortCombo.get(), num_device= int(self.num_device_entry.get()), baudrate=self.baudCombo.get())
                    self.motorName = self.motorNameCombo.get()
                else:
                    self.motor = Motor(port=self.myPortCombo.get(), baudrate=self.baudCombo.get())
                    self.motorName = self.motorNameCombo.get()
                #	    self.vm = videoInstron(self.ser)
                print 'connection'
                self.location.config(state='normal')
                self.moveButton.config(state='normal')
                self.recoilButton.config(state='normal')
                self.advanceButton.config(state='normal')
                self.moveZeroButton.config(state='normal')
                self.defineZeroButton.config(state='normal')
                self.stopMTButton.config(state='normal')
                self.stopVMButton.config(state='normal')

            except Exception as e:
                print ' Connection error:', e

    def callback(self, *args):
        print args
        print 'test: ', self.motorNameCombo.get().capitalize()
        if self.motorNameCombo.get().capitalize() == "Oriental":
            self.num_device_label.grid(row=4, column=0, sticky="sw", padx=10, pady=10)
            self.num_device_entry.grid(row=4, column=1, sticky="sw", padx=10, pady=10)
        else:
            self.num_device_entry.grid_forget()
            self.num_device_label.grid_forget()

    def defineZero(self):
        if self.motorName == "CmDrive":
            self.motor.ser.close()
            self.motor.ser.open()
            self.motor.ser.write('P=0\r')
            # self.motor.ser.readline()
            self.motor.ser.close()
        else:
            try:
                self.motor.actuator.set_home()
            except NotImplementedError:
                print "Not implemented yet."

    def reset_servostar(self):
        self.motor.reset()

    # function to examine the location of the motor
    def examineLocation(self):
        location = self.motor.sensor.get_position()
        self.Resultat.set(location)

    # function to move home
    def moveZero(self):
        self.motor.actuator.move_home()
        print 'moving home'

    # function to apply a motion task 
    def move(self):
        if self.motionType.get() == "" or self.position.get() == "":
            print "one of the entry is empty"
        else:
            if self.motorName == "CmDrive":
                if self.motionType.get() == '1':
                    self.motor.actuator.set_position(int(self.position.get()), None, 'absolute')
                    print 'MA mode'
                else:
                    self.motor.actuator.set_position(int(self.position.get()), None, 'relative')
                    print 'MR mode'
            else:
                self.motor.actuator.set_position(int(self.position.get()), int(self.speed.get()))
    # function to advance the motor on velocity mode
    def advance(self):
        if self.speedVar.get() == "":
            print 'choose velocity'
        else:
            self.motor.actuator.set_speed(int(self.speedVar.get()))
            print('the motor goes up with speed=%i' % int(self.speedVar.get()))

    # function to recoil the motor on velocity mode
    def recoil(self):
        if self.speedVar.get() == "":
            print 'choose velocity'
        else:
            self.motor.actuator.set_speed(-int(self.speedVar.get()))
            print('the motor goes down with speed=%i' % int(self.speedVar.get()))

    # function to stop a motion
    def stopMotion(self):
        self.motor.stop()
        print ' Motion has been stoped'


if __name__ == '__main__':
    root = Tix.Tk()
    interface = Interface(root)
    interface.mainloop()
    interface.destroy()

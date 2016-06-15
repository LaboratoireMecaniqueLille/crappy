# -*- coding:utf-8 -*-
import Tix
from Tkinter import *
import os
import time
import tkFont
import tkFileDialog
from _meta import MasterBlock


class Interface(Frame, MasterBlock):
    def __init__(self, root, VariateurTribo, comediOut):  # ,link1,link2):
        super(Interface, self).__init__()
        try:
            Frame.__init__(self, root, width=1000, height=1000)  # ,**kwargs)
            self.root = root
            self.root.title("Interface Tribomètre")
            police = tkFont.Font(self, size=15, family='Serif')
            self.t0 = 0
            self.init = False
            self.EnESSAI = False
            self.filepath = os.getcwd()
            self.mode = 'Mode Position'
            self.VariateurTribo = VariateurTribo
            self.comediOut = comediOut
            self.comediOut.off()
            self.comediOut.set_cmd(0)
            self.manualAuto = 'MANUEL'

            self.SpeedMin = 0
            self.SpeedMax = 1000

            self.ForceMin = 0
            self.ForceMax = 1000

            self.PositionMin = -6000
            self.PositionMax = 80000

            #### FRAME DECLARATION ####
            frameSave = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            frameManualAuto = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            frameTitleMode = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)

            # Manual
            self.frameSpeedForcePosition = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            self.frameSpeed = Frame(self.frameSpeedForcePosition, width=200, borderwidth=2, relief=GROOVE)
            self.frameForcePos = Frame(self.frameSpeedForcePosition, width=200, borderwidth=2, relief=GROOVE)

            self.frameData = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            self.frameInitialisation = Frame(self.frameData, width=200, borderwidth=2, relief=GROOVE)
            self.frameMode = Frame(self.frameData, width=200, borderwidth=2, relief=GROOVE)

            # Auto

            self.frameProperties = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            self.frameInformations = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)
            self.frameStart = Frame(self.root, width=400, borderwidth=2, relief=GROOVE)

            #### TITLE MODE DECLARATION ####

            self.TitleModeLabel = Label(frameTitleMode, text="MODE MANUEL", width=15)

            #### DATA RECORD DECLARATION ####
            # DataRecording
            self.pathSelectButton = Button(frameSave, text="Browse...", command=self.askdirectory)

            self.StartRecordDataButton = Button(frameSave, text="StartRecordData", command=self.go)
            self.StopRecordDataButton = Button(frameSave, text="StopRecordData", command=self.stop)
            self.filepathVar = Tix.StringVar()
            self.dirEntry = Entry(frameSave, textvariable=self.filepathVar, width=55)
            self.filepathVar.set(self.filepath)

            # defining options for opening a directory
            self.dir_opt = options = {}
            options['mustexist'] = True
            options['parent'] = self.root

            #### CHANGING MANUAL AUTO DECLARATION ####

            self.ChangeModeButton = Button(frameManualAuto, text='AUTO', command=self.changeManualAuto)

            #### MANUAL MODE DECLARATION ####


            # Speed Force and position of actuator
            self.ActuatorSpeed = Label(self.frameSpeed, text="Vitesse Broche (tr/min)")

            self.ActuatorForce = Label(self.frameForcePos, text="Effort Patin (N)")

            self.ActuatorPosition = Label(self.frameForcePos, text="Position platine (mm)")

            self.ActuatorPos = Label(self.frameForcePos, text="Position platine (mm)")

            self.ActuatorPosLabel = Label(self.frameForcePos, text="0", width=8)

            # self.ActuatorPositionActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.ActuatorPosUpdate())

            def ActuatorSpeedUpdate(var):
                if var < self.SpeedMin:
                    var = self.SpeedMin
                    self.ActuatorSpeedVar.set(self.SpeedMin)
                elif var > self.SpeedMax:
                    var = self.SpeedMax
                    self.ActuatorSpeedVar.set(self.SpeedMax)
                elif self.SpeedMin <= var <= self.SpeedMax:
                    self.comediOut.set_cmd(float(var))
                    print 'Vitesse demandée = ' + str(var)

            def ActuatorPositionUpdate(var):
                if var < self.PositionMin:
                    var = self.PositionMin
                    self.ActuatorPositionVar.set(self.PositionMin)
                elif var > self.PositionMax:
                    self.ActuatorPositionVar.set(self.PositionMax)
                    var = self.PositionMax
                elif self.PositionMin <= var <= self.PositionMax:
                    self.VariateurTribo.actuator.go_position(-int(var))
                    print 'Position demandée = ' + str(var)

            def ActuatorForceUpdate(var):
                if var < self.ForceMin:
                    self.ActuatorForceVar.set(self.ForceMin)
                    var = self.ForceMin
                elif var > self.ForceMax:
                    self.ActuatorForceVar.set(self.ForceMax)
                    var = self.ForceMax
                elif self.ForceMin <= var <= self.ForceMax:
                    self.VariateurTribo.actuator.go_effort(int(var) * 7.94)
                    print 'Force demandée = ' + str(var)

            # CONTROLS OF SETPOINT WITH SPINBOX
            self.ActuatorSpeedVar = Tix.IntVar()
            self.ActuatorSpeedSpinbox = Spinbox(self.frameSpeed, from_=self.SpeedMin, to=self.SpeedMax, increment=1,
                                                textvariable=self.ActuatorSpeedVar)
            self.ActuatorSpeedSpinbox.bind('<Return>', lambda event: ActuatorSpeedUpdate(self.ActuatorSpeedVar.get()))

            self.ActuatorForceVar = Tix.IntVar()
            self.ActuatorForceSpinbox = Spinbox(self.frameForcePos, from_=self.ForceMin, to=self.ForceMax, increment=1,
                                                textvariable=self.ActuatorForceVar)
            self.ActuatorForceSpinbox.bind('<Return>', lambda event: ActuatorForceUpdate(self.ActuatorForceVar.get()))

            self.ActuatorPositionVar = Tix.IntVar()
            self.ActuatorPositionSpinbox = Spinbox(self.frameForcePos, from_=self.PositionMin, to=self.PositionMax,
                                                   increment=100, textvariable=self.ActuatorPositionVar)
            self.ActuatorPositionSpinbox.bind('<Return>',
                                              lambda event: ActuatorPositionUpdate(self.ActuatorPositionVar.get()))

            # init Declaration
            self.InitButton = Button(self.frameInitialisation, text='Initialisation Variateur',
                                     command=self.initialisation)
            self.InitStatus = Label(self.frameInitialisation, text='Non initialisé')

            # Mode Declaration
            self.ModeButton = Button(self.frameMode, text='Changer Mode', command=self.changeMode)
            self.ModeLabel = Label(self.frameMode, text=self.mode)

            #### AUTO MODE DECLARATION ####

            self.MaxSpeedVar = Tix.IntVar()
            self.MaxSpeedVar.set(1000)
            self.MaxSpeedPropertyLabel = Label(self.frameProperties, text='Vitesse Max (tr/min)')
            self.MaxSpeedPropertyEntry = Entry(self.frameProperties, textvariable=self.MaxSpeedVar)

            self.MinSpeedVar = Tix.IntVar()
            self.MinSpeedPropertyLabel = Label(self.frameProperties, text='Vitesse Min (tr/min)')
            self.MinSpeedPropertyEntry = Entry(self.frameProperties, textvariable=self.MinSpeedVar)

            self.ForceVar = Tix.IntVar()
            self.ForceVar.set(300)
            self.ForcePropertyLabel = Label(self.frameProperties, text='Effort de freinage (N)')
            self.ForcePropertyEntry = Entry(self.frameProperties, textvariable=self.ForceVar)

            self.SimulatedInertia = Tix.StringVar()
            self.SimulatedInertia.set(3)
            self.SimulatedInertiaLabel = Label(self.frameProperties, text='Inertie simulée (kg.m²)')
            self.SimulatedInertiaEntry = Entry(self.frameProperties, textvariable=self.SimulatedInertia)

            self.cycleNumber = Tix.IntVar()
            self.cycleNumber.set(1)
            self.CycleNumberLabel = Label(self.frameProperties, text='Nombre de cycles')
            self.CycleNumberEntry = Entry(self.frameProperties, textvariable=self.cycleNumber)

            self.StartExperimentButton = Button(self.frameProperties, text="Démarrer l'essai",
                                                command=self.startExperiment)
            self.AbortExperimentButton = Button(self.frameProperties, text="Arrêter l'essai",
                                                command=self.stopExperiment)

            self.Informations = Label(self.frameInformations, text='Pret')

            #### POSITIONNING

            frameTitleMode.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            self.TitleModeLabel.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            ####
            frameSave.grid(row=2, column=1, sticky="w", padx=10, pady=10)
            # Inside FrameSave
            self.pathSelectButton.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            self.dirEntry.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            self.StartRecordDataButton.grid(row=2, column=0, sticky="w", padx=10, pady=10)
            self.StopRecordDataButton.grid(row=2, column=1, sticky="w", padx=10, pady=10)

            ####
            self.frameSpeedForcePosition.grid(row=3, column=1, sticky="w", padx=10, pady=10)
            # Inside frameSpeedForcePosition
            self.frameSpeed.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            # Inside frameSpeed
            self.ActuatorSpeed.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            self.ActuatorSpeedSpinbox.grid(row=1, column=2, sticky="w", padx=10, pady=10)

            self.frameForcePos.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            # Inside frameForcePos
            self.ActuatorForce.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            self.ActuatorForceSpinbox.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            self.ActuatorForce.grid_remove()
            self.ActuatorForceSpinbox.grid_remove()

            # ActuatorPositionButton
            self.ActuatorPosition.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            self.ActuatorPositionSpinbox.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            self.ActuatorPos.grid(row=2, column=1, sticky="w", padx=10, pady=10)
            self.ActuatorPosLabel.grid(row=2, column=2, sticky="w", padx=10, pady=10)
            self.ActuatorPosition.grid_remove()
            self.ActuatorPositionSpinbox.grid_remove()
            self.ActuatorPos.grid_remove()
            self.ActuatorPosLabel.grid_remove()

            ####
            self.frameData.grid(row=4, column=1, sticky="w", padx=10, pady=10)
            # Inside frameData
            self.frameInitialisation.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            # inside frameInitialisation
            self.InitButton.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            self.InitStatus.grid(row=1, column=1, sticky="w", padx=10, pady=10)

            self.frameMode.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            # Inside frameMode
            self.ModeButton.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            self.ModeLabel.grid(row=1, column=1, sticky="w", padx=10, pady=10)

            ####
            frameManualAuto.grid(row=5, column=1, sticky="w", padx=10, pady=10)
            # Inside frameManualAuto
            self.ChangeModeButton.grid(row=1, column=0, sticky="w", padx=10, pady=10)

            self.frameProperties.grid(row=3, column=1, sticky="w", padx=10, pady=10)
            # Inside frameProperties
            self.MaxSpeedPropertyLabel.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            self.MaxSpeedPropertyEntry.grid(row=1, column=1, sticky="w", padx=10, pady=10)

            self.MinSpeedPropertyLabel.grid(row=2, column=0, sticky="w", padx=10, pady=10)
            self.MinSpeedPropertyEntry.grid(row=2, column=1, sticky="w", padx=10, pady=10)

            self.ForcePropertyLabel.grid(row=3, column=0, sticky="w", padx=10, pady=10)
            self.ForcePropertyEntry.grid(row=3, column=1, sticky="w", padx=10, pady=10)

            self.SimulatedInertiaLabel.grid(row=4, column=0, sticky="w", padx=10, pady=10)
            self.SimulatedInertiaEntry.grid(row=4, column=1, sticky="w", padx=10, pady=10)

            self.CycleNumberLabel.grid(row=5, column=0, sticky="w", padx=10, pady=10)
            self.CycleNumberEntry.grid(row=5, column=1, sticky="w", padx=10, pady=10)

            self.StartExperimentButton.grid(row=3, column=2, sticky="w", padx=10, pady=10)

            self.frameInformations.grid(row=4, column=1, sticky="w", padx=10, pady=10)
            # Inside frameInformations
            self.Informations.grid(row=1, column=0, sticky="w", padx=10, pady=10)

            self.frameProperties.grid_remove()
            # Inside frameProperties
            self.MaxSpeedPropertyLabel.grid_remove()
            self.MaxSpeedPropertyEntry.grid_remove()

            self.MinSpeedPropertyLabel.grid_remove()
            self.MinSpeedPropertyEntry.grid_remove()

            self.ForcePropertyLabel.grid_remove()
            self.ForcePropertyEntry.grid_remove()

            self.SimulatedInertiaLabel.grid_remove()
            self.SimulatedInertiaEntry.grid_remove()

            self.CycleNumberLabel.grid_remove()
            self.CycleNumberEntry.grid_remove()

            self.StartExperimentButton.grid_remove()

            self.frameInformations.grid_remove()
            # Inside frameInformations
            self.Informations.grid_remove()

        # self.add_input(link1)
        # self.add_output(link2)
        except Exception as e:
            print e

    def main(self):
        self.mainloop()
        self.outputs[0].send(0)

    def changeManualAuto(self):

        if self.manualAuto == "MANUEL" and self.init:  # passage mode AUTO
            ####
            self.frameSpeedForcePosition.grid_remove()
            # Inside frameSpeedForcePosition
            self.frameSpeed.grid_remove()
            # Inside frameSpeed
            self.ActuatorSpeed.grid_remove()
            self.ActuatorSpeedSpinbox.grid_remove()

            self.frameForcePos.grid_remove()
            # Inside frameForcePos
            self.ActuatorForce.grid_remove()
            self.ActuatorForceSpinbox.grid_remove()

            # ActuatorPositionButton
            self.ActuatorPosition.grid_remove()
            self.ActuatorPositionSpinbox.grid_remove()
            self.ActuatorPos.grid_remove()
            self.ActuatorPosLabel.grid_remove()

            ####
            self.frameData.grid_remove()
            # Inside frameData
            self.frameInitialisation.grid_remove()
            # inside frameInitialisation
            self.InitButton.grid_remove()
            self.InitStatus.grid_remove()

            self.frameMode.grid_remove()
            # Inside frameMode
            self.ModeButton.grid_remove()
            self.ModeLabel.grid_remove()

            self.ChangeModeButton.configure(text="MANUEL")
            self.TitleModeLabel.configure(text="MODE AUTO")
            self.manualAuto = "AUTO"

            self.frameProperties.grid()
            self.MaxSpeedPropertyLabel.grid()
            self.MaxSpeedPropertyEntry.grid()

            self.MinSpeedPropertyLabel.grid()
            self.MinSpeedPropertyEntry.grid()

            self.ForcePropertyLabel.grid()
            self.ForcePropertyEntry.grid()

            self.SimulatedInertiaLabel.grid()
            self.SimulatedInertiaEntry.grid()

            self.CycleNumberLabel.grid()
            self.CycleNumberEntry.grid()

            self.StartExperimentButton.grid()

            self.frameInformations.grid()
            self.Informations.grid()




        elif self.manualAuto == "AUTO" and self.EnESSAI is False:  # passage au mode MANUEL

            self.frameProperties.grid_remove()
            # Inside frameProperties
            self.MaxSpeedPropertyLabel.grid_remove()
            self.MaxSpeedPropertyEntry.grid_remove()

            self.MinSpeedPropertyLabel.grid_remove()
            self.MinSpeedPropertyEntry.grid_remove()

            self.ForcePropertyLabel.grid_remove()
            self.ForcePropertyEntry.grid_remove()

            self.SimulatedInertiaLabel.grid_remove()
            self.SimulatedInertiaEntry.grid_remove()

            self.CycleNumberLabel.grid_remove()
            self.CycleNumberEntry.grid_remove()

            self.StartExperimentButton.grid_remove()

            self.frameInformations.grid_remove()
            # Inside frameInformations
            self.Informations.grid_remove()

            self.ChangeModeButton.configure(text="AUTO")
            self.TitleModeLabel.configure(text="MODE MANUEL")
            self.manualAuto = "MANUEL"

            self.frameSpeedForcePosition.grid()
            # Inside frameSpeedForcePosition
            self.frameSpeed.grid()
            # Inside frameSpeed
            self.ActuatorSpeed.grid()
            self.ActuatorSpeedSpinbox.grid()

            self.frameForcePos.grid()
            # Inside frameForcePos
            self.ActuatorForce.grid()
            self.ActuatorForceSpinbox.grid()

            # ActuatorPositionButton
            self.ActuatorPosition.grid()
            self.ActuatorPositionSpinbox.grid()
            self.ActuatorPos.grid()
            self.ActuatorPosLabel.grid()

            ####
            self.frameData.grid()
            # Inside frameData
            self.frameInitialisation.grid()
            # inside frameInitialisation
            self.InitButton.grid()
            self.InitStatus.grid()

            self.frameMode.grid()
            # Inside frameMode
            self.ModeButton.grid()
            self.ModeLabel.grid()
        else:
            print "Initialisez d'abord"

    def startExperiment(self):
        print 'top'
        try:
            cycle = self.cycleNumber.get()
            while cycle > 0:
                self.outputs[0].send(1)
                tStart = time.time()
                tAfter = time.time()
                print tAfter - tStart
                while tAfter - tStart < 0.5:
                    # print 'top2bis'
                    value = self.inputs[0].recv()
                    # print 'top2ter'
                    tAfter = time.time()

                print value['Vit']
                while float(value['Vit']) <= self.MaxSpeedVar.get() * 9.0 / 10.0:
                    self.comediOut.set_cmd(self.MaxSpeedVar.get())
                    value = self.inputs[0].recv()

                tStart = time.time()
                tAfter = time.time()

                while tAfter - tStart < 2:
                    value = self.inputs[0].recv()
                    tAfter = time.time()

                tStart = time.time()
                tAfter = time.time()
                print 'top3'
                C0 = float(value['Couple'])
                V0 = float(value['Vit'])
                V1 = V0
                print 'top4'
                self.comediOut.on()
                self.VariateurTribo.actuator.set_mode_analog()
                print 'force demandée=', int(self.ForceVar.get())
                self.VariateurTribo.actuator.go_effort(int(self.ForceVar.get()) * 10)
                # self.VariateurTribo.actuator.set_mode_position()
                # self.VariateurTribo.actuator.go_position(-18000)


                while V1 > self.MinSpeedVar.get():  # Freinage
                    tStart = tAfter
                    tAfter = time.time()
                    deltaVit = (tAfter - tStart) * (float(value['Couple']) - C0) / float(self.SimulatedInertia.get())
                    V1 = V1 - deltaVit

                    self.comediOut.set_cmd(V1)
                    value = self.inputs[0].recv()

                self.VariateurTribo.actuator.set_mode_position()
                self.VariateurTribo.actuator.go_position(0)
                self.comediOut.set_cmd(0)
                self.outputs[0].send(0)

                cycle -= 1

        except NameError:
            self.comediOut.set_cmd(0)
        # pass
        # if self.EnESSAI is False:
        # self.EnESSAI=True
        # try:
        # while int(self.inputs.recv()['Vit'])<= int(self.MaxSpeedVar.get()):
        # self.comediOut.set_cmd(float(self.MaxSpeedVar))
        ##pass
        # time.sleep(1)
        # self.comediOut.set_cmd(0)

        # except:
        # pass

        ##self.EnESSAI=False
        # else:
        # pass

    def stopExperiment(self):
        raise NameError

    # pass
    # raise une exception pour le startexperiment et arrete la broche et fait un homing

    def pathSelect(self):
        d = Tix.DirSelectBox(master=self.root, command=self.print_selected)
        d.popup()

    def askdirectory(self):
        """
        Returns a selected directoryname.
        """
        self.filepath = tkFileDialog.asksaveasfilename()
        self.filepathVar.set(self.filepath)

    def __str__(self):
        return "RecordDataFile: {0}\n".format(self.filepath)

    def getInfo(self):
        return self.filepath

    def ActuatorPosUpdate(self):
        try:
            self.VariateurTribo.sensor.clear()
            self.ActuatorPosLabel.configure(text=self.VariateurTribo.sensor.read_position())
            self.root.after(20, self.ActuatorPositionUpdate)
        except:
            self.ActuatorPosUpdate()
            print'error reading position'

    def StartRecordData(self):
        self.t = str(time.time() - self.t0)
        RecordData(self.t, self.ActuatorPositionValue, self.LoadValue, self.filepath)
        self.cpt += 1
        self.RecordDataNumberLabel.configure(text=self.cpt)
        if self.flag == 1:
            self.root.after(1000, self.StartRecordData)

    def go(self):
        if self.flag == 0:
            self.flag = 1
            self.t0 = time.time()  # Initialisation du temps
            self.StartRecordData()
        print "RecordDataStart"

    def stop(self):
        self.flag = 0
        print "RecordDataStop"

    def changeMode(self):
        if self.mode == 'Mode Position' and self.init:  # let's go to force mode
            self.mode = 'Mode Effort'
            self.ActuatorPosition.grid_remove()
            self.ActuatorPositionSpinbox.grid_remove()
            self.ActuatorPos.grid_remove()
            self.ActuatorPosLabel.grid_remove()
            self.ActuatorForce.grid()
            self.ActuatorForceSpinbox.grid()
            # self.ActuatorEffort.grid()
            # self.ActuatorEffortLabel.grid()
            self.comediOut.on()
            self.VariateurTribo.actuator.set_mode_analog()


        elif self.mode == 'Mode Effort' and self.init:  # let's go to position mode
            self.mode = 'Mode Position'
            self.ActuatorPosition.grid()
            self.ActuatorPositionSpinbox.grid()
            self.ActuatorPos.grid()
            self.ActuatorPosLabel.grid()
            self.ActuatorForce.grid_remove()
            self.ActuatorForceSpinbox.grid_remove()
            # self.ActuatorEffort.grid_remove()
            # self.ActuatorEffortLabel.grid_remove()
            self.VariateurTribo.actuator.set_mode_position()
            self.comediOut.off()
            self.ActuatorPositionVar.set(self.VariateurTribo.sensor.read_position())

        self.ModeLabel.configure(text=self.mode)

    def initialisation(self):
        self.outputs[0].send(0)
        self.init = True
        if self.mode == 'Mode Position':
            self.ActuatorPosition.grid()
            self.ActuatorPositionSpinbox.grid()
            self.ActuatorPos.grid()
            self.ActuatorPosLabel.grid()
        self.VariateurTribo.sensor.clear()
        self.VariateurTribo.actuator.initialisation()
        # self.VariateurTribo.actuator.set_mode_position()
        # self.ActuatorPosLabel.configure(text=str(self.VariateurTribo.sensor.read_position()))
        time.sleep(1)
        self.VariateurTribo.sensor.clear()
        time.sleep(1)
        self.InitStatus.configure(text='Initialisé')

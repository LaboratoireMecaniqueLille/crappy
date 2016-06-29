import os
import time

class Interface(Frame, MasterBlock):
    def __init__(self, root, VariateurTribo):  # ,link1,link2):
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
            #self.comediOut = comediOut
            #self.comediOut.off()
            #self.comediOut.set_cmd(0)
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
            
            #### TITLE MODE DECLARATION ####

            self.TitleModeLabel = Label(frameTitleMode, text="MODE MANUEL", width=15)
            
            #### MANUAL MODE DECLARATION ####


            # Speed Force and position of actuator
            self.ActuatorSpeed = Label(self.frameSpeed, text="Vitesse Broche (tr/min)")

            self.ActuatorForce = Label(self.frameForcePos, text="Effort Patin (N)")

            self.ActuatorPosition = Label(self.frameForcePos, text="Position platine (mm)")

            self.ActuatorPos = Label(self.frameForcePos, text="Position platine (mm)")

            self.ActuatorPosLabel = Label(self.frameForcePos, text="0", width=8)
            
            #def ActuatorSpeedUpdate(var):
                #if var < self.SpeedMin:
                    #var = self.SpeedMin
                    #self.ActuatorSpeedVar.set(self.SpeedMin)
                #elif var > self.SpeedMax:
                    #var = self.SpeedMax
                    #self.ActuatorSpeedVar.set(self.SpeedMax)
                #elif self.SpeedMin <= var <= self.SpeedMax:
                    #self.comediOut.set_cmd(float(var))
                    #print 'Vitesse demandée = ' + str(var)
                    
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

            #def ActuatorForceUpdate(var):
                #if var < self.ForceMin:
                    #self.ActuatorForceVar.set(self.ForceMin)
                    #var = self.ForceMin
                #elif var > self.ForceMax:
                    #self.ActuatorForceVar.set(self.ForceMax)
                    #var = self.ForceMax
                #elif self.ForceMin <= var <= self.ForceMax:
                    #self.VariateurTribo.actuator.go_effort(int(var) * 7.94)
                    #print 'Force demandée = ' + str(var)
                    
            # CONTROLS OF SETPOINT WITH SPINBOX
            #self.ActuatorSpeedVar = Tix.IntVar()
            #self.ActuatorSpeedSpinbox = Spinbox(self.frameSpeed, from_=self.SpeedMin, to=self.SpeedMax, increment=1,
                                                #textvariable=self.ActuatorSpeedVar)
            #self.ActuatorSpeedSpinbox.bind('<Return>', lambda event: ActuatorSpeedUpdate(self.ActuatorSpeedVar.get()))

            #self.ActuatorForceVar = Tix.IntVar()
            #self.ActuatorForceSpinbox = Spinbox(self.frameForcePos, from_=self.ForceMin, to=self.ForceMax, increment=1,
                                                #textvariable=self.ActuatorForceVar)
            #self.ActuatorForceSpinbox.bind('<Return>', lambda event: ActuatorForceUpdate(self.ActuatorForceVar.get()))

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
            #self.ActuatorSpeed.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            #self.ActuatorSpeedSpinbox.grid(row=1, column=2, sticky="w", padx=10, pady=10)

            self.frameForcePos.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            # Inside frameForcePos
            #self.ActuatorForce.grid(row=1, column=1, sticky="w", padx=10, pady=10)
            #self.ActuatorForceSpinbox.grid(row=1, column=2, sticky="w", padx=10, pady=10)
            #self.ActuatorForce.grid_remove()
            #self.ActuatorForceSpinbox.grid_remove()

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
            
	else:
            print "Initialisez d'abord"
            
    def ActuatorPosUpdate(self):
        try:
            self.VariateurTribo.sensor.clear()
            self.ActuatorPosLabel.configure(text=self.VariateurTribo.sensor.get_position())
            self.root.after(20, self.ActuatorPositionUpdate)
        except:
            self.ActuatorPosUpdate()
            print'error reading position'
       
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
        # self.ActuatorPosLabel.configure(text=str(self.VariateurTribo.sensor.get_position()))
        time.sleep(1)
        self.VariateurTribo.sensor.clear()
        time.sleep(1)
        self.InitStatus.configure(text='Initialisé')

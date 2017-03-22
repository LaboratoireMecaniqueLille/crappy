# -*- coding:utf-8 -*-
from .masterblock import MasterBlock
import Tix
from Tkinter import *
import os
import time
import tkFont
import tkFileDialog
import crappy
from collections import OrderedDict
########INTEGRATION DU GRAPHE DANS L'INTERFACE (putain, j'en ai chié)
from matplotlib.lines import Line2D

#matplotlib.use("TkAgg") # Seems useless and raises a warning
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt

import Tkinter as tk
import tkMessageBox


########

class InterfaceTomo4D(MasterBlock, Frame):
    # def __init__(self, root,Actuator,TransX,TransY,Load,position_pipe_recv,force_pipe_recv, **kwargs):
    def __init__(self, root, technical, TransX, TransY, Load, **kwargs):
        Frame.__init__(self, root, width=1000, height=1000, **kwargs)
        MasterBlock.__init__(self)
        print "interface !",os.getpid()

        self.root = root
        self.root.title("Greg's interface")
        police = tkFont.Font(self, size=15, family='Time')

        self.t = 0
        self.tbuffered = []
        self.ActuatorPositionValue = 0
        self.ActuatorPositionValuebuffered = []
        self.LoadValue0 = 0
	self.LoadValue = 0
	self.LoadValuebuffered = []
	self.flag = 0
	self.cpt = 0
	self.PositionTare = 0
	self.LoadTare = 0
	self.t0 = 0
	self.PIDflag = 0
	self.recursive_func_log =  []
	self.filepath = os.getcwd()
	self.pid_on = True
	
	self.technical = technical
	self.TransX = TransX
	self.TransY = TransY
	self.Load = Load
	self.Speed = 0.5 #Starting speed at 0.5 mm/s
	self.SpeedPID = 0.0
	
	self.Exx = 0
	self.Eyy = 0
	self.Px = 0
	self.Py = 0
	
	self.checkvar = 0
	self.PIDTitle_level = tk.StringVar()
	#self.var = 0
	
	#FrameDeclaration
        frameData = Frame(self.root, width=400, borderwidth=2,relief=GROOVE) # create a frame in canvas(p1)
        frameActuatorInfo = Frame(self.root, width=400, borderwidth=2,relief=GROOVE)
        self.framePID=Frame(self.root, width=400, borderwidth=2,relief=GROOVE)
        frameActuator = Frame(frameActuatorInfo, width=400, borderwidth=2,relief=GROOVE) # create a frame in canvas(p1)
        frameInfo = Frame(frameActuatorInfo, width=400, borderwidth=2,relief=GROOVE)
        frameLoad = Frame(frameInfo, width=200, borderwidth=2,relief=GROOVE)
        frameActuatorPosition = Frame(frameInfo, width=200, borderwidth=2,relief=GROOVE) # create a frame in canvas(p1)
        frameTransXY = Frame(self.root, width=400, borderwidth=2,relief=GROOVE)
        self.frameGraph = Frame(self.root, width=400, borderwidth=2,relief=GROOVE)
        frameExtenso = Frame(frameInfo, width=800, borderwidth=2,relief=GROOVE)
        
        self.var = Tix.StringVar()
        self.portLabel = Label(frameData, textvariable= self.var ) # create a label 
        self.cameraSelection= Tix.StringVar() # create a variable, it will contain port selection 

        
        #ReadInput0 - Bouton fantôme pour lire les entrées (Displacement and Load)
        self.ReadInputActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.ReadInput())
        #ReadInput1 - Bouton fantôme pour lire les entrées (Displacement and Load)
        #self.ReadExtensoActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.ReadExtenso())
        #SendOutput - Bouton fantôme pour envoyer les sorties (speed, flagrecord, pid options)
        self.SendOutputActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.SendOutput())
        #ActuatorPositionDeclaration
        self.ActuatorPositionTitleLabel = Label(frameActuatorPosition, text="ActuatorPosition (mm)")
	self.ActuatorPositionLabel = Label(frameActuatorPosition, text="0", width = 8)
	self.ActuatorPositionActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.ActuatorPositionUpdate())
	#LoadDeclaration
	self.LoadTitle = Label(frameLoad, text="Load (N)") #create a label
	self.LoadLabel = Label(frameLoad, text="0",width = 8)
	self.LoadActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.LoadUpdate())
	#ExtensoDeclaration
        self.ExxTitleLabel = Label(frameExtenso, text="Exx")
	self.ExxLabel = Label(frameExtenso, text="0", width = 8)
	self.PxTitleLabel = Label(frameExtenso, text="Px")
	self.PxLabel = Label(frameExtenso, text="0", width = 8)
	self.EyyTitleLabel = Label(frameExtenso, text="Eyy")
	self.EyyLabel = Label(frameExtenso, text="0", width = 8)
	self.PyTitleLabel = Label(frameExtenso, text="Py")
	self.PyLabel = Label(frameExtenso, text="0", width = 8)
	#self.ExtensoActualisationGhostButton = Button(self.root,text="+1",bg="yellow",command=self.ExtensoUpdate())
	#Graph
	self.GraphGhostButton = Button(self.root,text="+1",bg="yellow",command=self.RealtimePloter())
	#PID
	#self.PIDTitle = Label(self.framePID, text="PID")
	self.PIDButtonStart = Button(self.framePID, text="Start PID", command = self.PIDgo)
	self.PIDButtonStop = Button(self.framePID, text="Stop PID", command = self.PIDstop)
	#Toggle PIDlabels enabled/disabled
	#self.PIDCheckButton = Checkbutton(self.framePID, state = ACTIVE,variable=self.checkvar,command=self.hide_PIDlabels)
	#HideButton
	self.PIDTitle = Button(self.framePID,text="PID",bg="green",command=self.change_button_level)
	

	#PIDspeed
	self.ActuatorSpeedPID = Label(self.framePID, text="Actuator Speed PID (mm/s)")
        self.ActuatorSpeedVarPID = Tix.StringVar()

        self.ActuatorSpeedPIDSpinbox= Spinbox(self.framePID, from_=0, to=1, increment=0.001, textvariable= self.ActuatorSpeedVarPID )
        self.ActuatorSpeedVarPID.trace('w', self.ActuatorSpeedUpdatePID)
	
	#PIDLoad
	self.LoadPID = Label(self.framePID, text="Load target value")
        self.LoadVarPID = Tix.StringVar()

        self.LoadPIDSpinbox= Spinbox(self.framePID, from_=-5000, to=5000, increment=100, textvariable= self.LoadVarPID)
        self.LoadVarPID.trace('w', self.LoadUpdatePID)
	

	#RecordDataDeclaration
	self.RecordDataNumberLabel = Label(frameData, text="0")
	self.StartRecordDataButton = Button(frameData, text="StartRecordData", command=self.go)
	self.StopRecordDataButton = Button(frameData, text="StopRecordData", command=self.stopRecord) 
        self.dirLabel = Label(frameData, text="RecordFileName:") #create a label
        self.filepathVar = Tix.StringVar()
        self.dirEntry = Entry(frameData, textvariable=self.filepathVar, width=55)
        self.filepathVar.set(self.filepath)

        # defining options for opening a directory
        self.dir_opt = options = {}
        options['mustexist'] = True
        options['parent'] = self.root

        # DataRecording
        self.pathSelectButton = Button(frameData, text="Browse...", command=self.askdirectory)
        
        ## ActuatorMoveButtons
        #self.ActuatorUp = Button(frameActuator, text="Up", command=self.technical.move_up)
        #self.ActuatorStop = Button(frameActuator, text="Stop", command=self.move_stop)
        #self.ActuatorDown = Button(frameActuator, text="Down", command=self.move_down)
        #self.ActuatorClearAlarm = Button(frameActuator, text="ClearAlarm", command=self.technical.clear_errors)
        #self.ActuatorPositionTareButton = Button(frameActuatorPosition, text="TarePosition",
                                                 #command=self.technical.actuator.set_home)
        #######Needed to make the dummy work, if IRL, use the lines above
        self.ActuatorUp = Button(frameActuator, text="Up", command=self.move_up)
        self.ActuatorStop = Button(frameActuator, text="Stop", command=self.move_stop)
        self.ActuatorDown = Button(frameActuator, text="Down", command=self.move_down)
        self.ActuatorClearAlarm = Button(frameActuator, text="ClearAlarm", command=self.technical.clear_errors)     
        self.ActuatorPositionTareButton = Button(frameActuatorPosition, text="TarePosition",
                                                 command=self.technical.set_home)
        
        
        
        
        # ActuatorSpeedButton
        self.ActuatorSpeed = Label(frameActuator, text="Actuator Speed (mm/s)")
        self.ActuatorSpeedVar = Tix.StringVar()

        self.ActuatorSpeedSpinbox = Spinbox(frameActuator, from_=0, to=1, increment=0.001,
                                            textvariable=self.ActuatorSpeedVar)
        #self.ActuatorSpeedVar.trace('w', self.ActuatorSpeedUpdate)

        # LoadTareButton
        self.LoadTareButton = Button(frameLoad, text="TareLoad", command=self.LoadTareAction)

        # LinearTranslationXMoveButtons
        self.TransXPos = Button(frameTransXY, text="TransX+", command=self.TransX.MoveUp)
        self.TransXStop = Button(frameTransXY, text="Stop", command=self.TransX.MoveStop)
        self.TransXNeg = Button(frameTransXY, text="TransX-", command=self.TransX.MoveDown)
        self.TransXClearAlarm = Button(frameTransXY, text="ClearAlarm-TransX", command=self.TransX.ClearAlarm)
        # LinearTranslationXMoveButtons
        self.TransYPos = Button(frameTransXY, text="TransY+", command=self.TransY.MoveUp)
        self.TransYStop = Button(frameTransXY, text="Stop", command=self.TransY.MoveStop)
        self.TransYNeg = Button(frameTransXY, text="TransY-", command=self.TransY.MoveDown)
        self.TransYClearAlarm = Button(frameTransXY, text="ClearAlarm-TransY", command=self.TransY.ClearAlarm)

        # LinearTranslationXSpeedButton
        self.TransXSpeed = Label(frameTransXY, text="TransX Speed [0,10]")
        self.TransXSpeedVar = Tix.StringVar()


        self.TransXSpeedSpinbox = Spinbox(frameTransXY, from_=0, to=10, increment=1, textvariable=self.TransXSpeedVar,
                                          width=4)
        self.TransXSpeedVar.trace('w', self.TransXSpeedUpdate)

        # LinearTranslationYSpeedButton
        self.TransYSpeed = Label(frameTransXY, text="TransY Speed [0,10]")
        self.TransYSpeedVar = Tix.StringVar()

        self.TransYSpeedSpinbox = Spinbox(frameTransXY, from_=0, to=10, increment=1, textvariable=self.TransYSpeedVar,
                                          width=4)
        self.TransYSpeedVar.trace('w', self.TransYSpeedUpdate)
        
        #POSITIONNING
        
        #FramePosition
        frameData.grid(row=1, column=1,sticky= "w", padx=1,pady=1) 
        frameActuatorInfo.grid(row=2,column=2,sticky= "w",padx=1,pady=1)
        frameActuator.grid(row=2,column=2,sticky= "w",padx=1,pady=1)
        frameInfo.grid(row=3,column=2,sticky= "w",padx=1,pady=1)
        frameActuatorPosition.grid(row=1,column=1,sticky= "w",padx=1,pady=1)
        frameLoad.grid(row=1,column=2,sticky= "w",padx=1,pady=1)
        frameTransXY.grid(row=3, column=1,sticky= "w",padx=1,pady=1)
        self.frameGraph.grid(row=2,column=1,sticky= "w",padx=1,pady=1)
        self.framePID.grid(row=1,column=2,sticky="w",padx=1,pady=1)
        frameExtenso.grid(row=4,column=1,sticky="w",padx=1,pady=1)
	
	#InsideInfoFrame
	#InsideActuatorPositionFrame
	self.ActuatorPositionTitleLabel.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
	self.ActuatorPositionLabel.grid(row=1, column=2, sticky= "w",padx=10,pady=10)
	self.ActuatorPositionTareButton.grid(row=2, column=2, sticky= "w",padx=10,pady=10)
	#InsideLoadFrame
	self.LoadTitle.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
	self.LoadLabel.grid(row=1, column=2, sticky= "w",padx=10,pady=10)
	self.LoadTareButton.grid(row=2, column=2, sticky= "w",padx=10,pady=10)
	
	#InsideExtensoFrame
	self.ExxTitleLabel.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
	self.ExxLabel.grid(row=1, column=2, sticky= "w",padx=10,pady=10)
	self.EyyTitleLabel.grid(row=2, column=1, sticky= "w",padx=10,pady=10)
	self.EyyLabel.grid(row=2, column=2, sticky= "w",padx=10,pady=10)
	self.PxTitleLabel.grid(row=3, column=1, sticky= "w",padx=10,pady=10)
	self.PxLabel.grid(row=3, column=2, sticky= "w",padx=10,pady=10)
	self.PyTitleLabel.grid(row=4, column=1, sticky= "w",padx=10,pady=10)
	self.PyLabel.grid(row=4, column=2, sticky= "w",padx=10,pady=10)
	
	#Inside PID frame
	#self.PIDTitle.grid(row=1,column=1,sticky="w",padx=10,pady=10,in_=self.framePID)
	self.PIDButtonStart.grid(row=2,column=1,sticky="w",padx=10,pady=10,in_=self.framePID)
	self.PIDButtonStop.grid(row=3,column=1,sticky="w",padx=10,pady=10)
	self.ActuatorSpeedPID.grid(row=2, column=2, sticky= "w",padx=10,pady=10,in_=self.framePID)
        self.ActuatorSpeedPIDSpinbox.grid(row=2, column=3, sticky= "w",padx=10,pady=10)
        self.LoadPID.grid(row=3, column=2, sticky= "w",padx=10,pady=10)
        self.LoadPIDSpinbox.grid(row=3, column=3, sticky= "w",padx=10,pady=10)
        #self.PIDCheckButton.grid(row=4,column=1,sticky="w",padx=10,pady=10)
        self.PIDTitle.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
        #self.PIDTitle_level.set(MSG_03)
		
	#InsideRecordDataFrame
        self.pathSelectButton.grid(row=1, column=0, sticky= "w",padx=10,pady=10)
        self.dirEntry.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
	self.StartRecordDataButton.grid(row=2, column=0, sticky= "w",padx=10,pady=10)
	self.StopRecordDataButton.grid(row=2, column=1, sticky= "w",padx=10,pady=10)
	self.RecordDataNumberLabel.grid(row=2, column=2, sticky= "w",padx=10,pady=10)	

	#InsideActuatorFrame
	#ActuatorSpeedButton
	self.ActuatorSpeed.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
        self.ActuatorSpeedSpinbox.grid(row=1, column=2, sticky= "w",padx=10,pady=10)
        #ActuatorMoveButtons
        self.ActuatorUp.grid(row=2, column=1, sticky= "w",padx=10,pady=10)
	self.ActuatorStop.grid(row=3, column=1, sticky= "w",padx=10,pady=10)
	self.ActuatorDown.grid(row=4, column=1, sticky= "w",padx=10,pady=10)
	self.ActuatorClearAlarm.grid(row=3, column=2, sticky= "w",padx=10,pady=10)
	
	#InsideTransXYFrame
	#LinearTranslationXMoveButtons
        self.TransXPos.grid(row=1, column=3, sticky= "w",padx=10,pady=10)
        self.TransXStop.grid(row=1, column=2, sticky= "w",padx=10,pady=10)
        self.TransXNeg.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
        self.TransXClearAlarm.grid(row=1, column=4, sticky= "w",padx=10,pady=10)
	self.TransXSpeed.grid(row=1, column=5, sticky= "w",padx=10,pady=10)
        self.TransXSpeedSpinbox.grid(row=1, column=6, sticky= "w",padx=10,pady=10)
        #LinearTranslationYMoveButtons
        self.TransYPos.grid(row=2, column=3, sticky= "w",padx=10,pady=10)
        self.TransYStop.grid(row=2, column=2, sticky= "w",padx=10,pady=10)
        self.TransYNeg.grid(row=2, column=1, sticky= "w",padx=10,pady=10)
        self.TransYClearAlarm.grid(row=2, column=4, sticky= "w",padx=10,pady=10)
        self.TransYSpeed.grid(row=2, column=5, sticky= "w",padx=10,pady=10)
        self.TransYSpeedSpinbox.grid(row=2, column=6, sticky= "w",padx=10,pady=10)
        
        self.graph() # Initialisation of dynamic graph in the GUI
    
    def move_down(self):
      #self.technical.actuator.set_speed(-float(self.ActuatorSpeedVar.get()))
      self.technical.set_speed(-float(self.ActuatorSpeedVar.get()))
    
    def move_stop(self):
      #self.technical.actuator.set_speed(0)
      self.technical.set_speed(0)
      
    def move_up(self):
      #self.technical.actuator.set_speed(float(self.ActuatorSpeedVar.get()))
      self.technical.set_speed(float(self.ActuatorSpeedVar.get()))

    def TransXSpeedUpdate(self, a, b, c):
	# print "TransXSpeed"
	self.TransX.Speed(float(self.TransXSpeedVar.get()))
	
    def TransYSpeedUpdate(self, a, b, c):
	# print "TransYSpeed"
	self.TransY.Speed(float(self.TransYSpeedVar.get()))
            
    def ActuatorSpeedUpdatePID(self, a, b, c):
	try:
	  self.SpeedPID = float(self.ActuatorSpeedVarPID.get())
	  #Actuator.Speed(self.SpeedPID)
	  #print self.SpeedPID speed
	except ValueError: 
	  print "SpeedPIDReadingErrorGI"
	  pass 
    
    def LoadUpdatePID(self, a, b, c):
	#print "ActuatorSpeed"
	try:
	  self.LoadPID = float(self.LoadVarPID.get())
	  print self.LoadPID
	except ValueError: 
	  print "LoadPIDReadingErrorGI"
	  pass 


###Dummy!!!!!	    
    def ActuatorSpeedUpdate(self, a, b, c):
	# print "ActuatorSpeed"
	try:
	    self.Speed = float(self.ActuatorSpeedVar.get())
	    self.technical.set_speed(self.Speed) #Dummy
	    print "GUISpeed", self.Speed
	    #self.technical.actuator.set_speed(self.Speed) #IRL
	except ValueError:
	    print "SpeedReadingErrorGI"
	    pass
	      
    def graph(self):
        self.GraphTitleLabel = Label(self.frameGraph,
                                     text="Displacement and load vs. time")  # Graph name and frame location
        self.GraphTitleLabel.grid(row=1, column=1, sticky="w", padx=10, pady=10)
        self.f = Figure(figsize=(6, 6), dpi=100)  # Figure inside canvas
        self.a = self.f.add_subplot(111)  # Subplot
        self.a2 = self.a.twinx()
        self.line1 = Line2D([], [],
                            color='black')  # Line2D is mandatory because line.set_data is not working otherwise (Fuck you matplotlib)
        self.line2 = Line2D([], [],
                            color='blue')  # Line2D is mandatory because line.set_data is not working otherwise (Fuck you matplotlib)
        self.a.add_line(self.line1)  # Link Line to subplot. Otherwise, nothing will happen. Life's so funny.
        self.a2.add_line(self.line2)  # Link Line to subplot. Otherwise, nothing will happen. Life's so funny.
        self.a.set_xlabel('time (s)')
        self.a.set_ylabel('Displacement (mm)')
        self.a2.set_ylabel('Load (N)')
        self.canvas = FigureCanvasTkAgg(self.f, self.frameGraph)  # self.frameGraph
        self.canvas.show()  # show the canvas
        self.canvas.get_tk_widget().grid(row=2, column=1, sticky="w", padx=10,
                                         pady=10)  # Positionnig canvas in frameGraph

    def RealtimePloter(self):
	  try:
	    # Updating data
	    self.tbuffered = self.tbuffered +[self.t]
	    self.ActuatorPositionValuebuffered = self.ActuatorPositionValuebuffered + [self.ActuatorPositionValue]
	    self.LoadValuebuffered = self.LoadValuebuffered + [self.LoadValue]
	    if len(self.tbuffered)>6000: #on n'affiche pas plus de 10 minutes ~ 60000 points
	      self.tbuffered = self.tbuffered[1::] +[self.t]
	      self.ActuatorPositionValuebuffered = self.ActuatorPositionValuebuffered[1:len(self.tbuffered)] + [self.ActuatorPositionValue]
	      self.LoadValuebuffered = self.LoadValuebuffered[1:len(self.tbuffered)] + [self.LoadValue]
	    self.line1.set_data([self.tbuffered],[self.ActuatorPositionValuebuffered])
	    self.line2.set_data([self.tbuffered],[self.LoadValuebuffered])
	    self.a.relim() #relimit plot axis
	    self.a2.relim()
	    self.a.autoscale_view(True,True,True) #rescale
	    self.a2.autoscale_view(True,True,True)
	    self.canvas.draw()#updating canvas
	  except AttributeError:
	    print "RealtimePloterAttributeError"
	  except TypeError:
	    print "RealtimePloterTypeError"
	  self.recursive_func_log.append(self.root.after(25,self.RealtimePloter))

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
        return self.filepathmove_stop

    def main(self):  # La fonction qui sert à faire vivre l'interface
        def on_closing():
            if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
                self.root.destroy()

        try:
            condition = 1
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            self.mainloop()
            #self.close_all_instances()
        except KeyboardInterrupt:
            #self.close_all_instances()
            crappy.stop()
            self.root.destroy()
        except Exception as e:
            print e
            #self.close_all_instances()
            crappy.stop()
            self.root.destroy()

    def stop(self):
        self.root.destroy()
        # self.proc.terminate()

    def ReadInput(self):
      try:
	self.data = self.inputs[0].recv()
	self.data= self.data.values()
	self.t, self.ActuatorPositionValue,self.LoadValue = self.data
	#print 'tinput',self.tframe
	#print 'datainput0',self.data
      except Exception as e: #if not init yetself.ActuatorSpeedPID
	print "ReadInputGIError",e
      self.recursive_func_log.append(self.root.after(10,self.ReadInput))
     
     
  ##############################################Extenso Reader#########################################
    #def ReadExtenso(self):
      #try:
	#self.dataExtenso = self.inputs[1].recv()
	##print self.dataExtenso
	#self.dataExtenso= self.dataExtenso.values()
	#self.Px,self.Py,self.Exx,self.Eyy = self.dataExtenso[1::]
	##print 'datainput1',self.Px
      #except Exception: #if not init yet
	#print "ReadInputExtensoGIError"
      #self.recursive_func_log.append(self.root.after(10,self.ReadExtenso))
      
    #def ExtensoUpdate(self):
      #try:
	#self.Exx = round(self.Exx,5)
	#self.Eyy = round(self.Eyy,5)
	##self.Px = round(self.Px,2) NE FONCTIONNE PAS PARCE QU'UN CONNARD A DECIDE DE DEFINIR Px et Py comme des char --> Gros connardos va
	#self.ExxLabel.configure(text=self.Exx)
	#self.EyyLabel.configure(text=self.Eyy)
	#self.PxLabel.configure(text=self.Px)
	#self.PyLabel.configure(text=self.Py)
      #except TypeError:
	#print "ExtensoUpdateErrorGI" #Se prévenir des erreur de lecture quand la readline fait de la merde
      #self.recursive_func_log.append(self.root.after(100,self.ExtensoUpdate))

###########################################################################################################"

    def SendOutput(self):
      try:
	#for savergui en mode parallèle, mais comme ça marche pas, on va bourriner
	#Array=OrderedDict(zip(['t(s)','RecordFlag','RecordPath','PIDflag','Vitesse'],[self.t,self.flag,self.filepath,self.PIDflag,self.Speed]))
	# for saverguiOneInput
	Array=OrderedDict(zip(['t(s)','position (mm)','F(N)','RecordFlag','RecordPath','PIDFlag','SpeedPID','LoadPID'],[self.t,self.ActuatorPositionValue,self.LoadValue,self.flag,self.filepath,self.PIDflag,self.SpeedPID,float(self.LoadVarPID.get())]))
	#Array=pd.DataFrame([[self.t,self.flag,self.filepath,self.PIDflag,self.Speed]],columns=['t(s)','RecordFlag','RecordPath','PIDflag','Vitesse'])
	for output in self.outputs:
            output.send(Array)
	#self.outputs[0].send(Array)
	#print Array
	#Saving points counter
	if self.flag ==1:
	  self.cpt +=1
	  self.RecordDataNumberLabel.configure(text=self.cpt)
	#print Array
      except AttributeError: #if not init yet
	pass
      self.recursive_func_log.append(self.root.after(100,self.SendOutput)) #Defini l'intervalle d'envoie des données vers l'extérieur
      
    def ActuatorPositionUpdate(self):
        try:
            self.ActuatorPositionValue = round(self.ActuatorPositionValue, 5)
            self.ActuatorPositionLabel.configure(text=self.ActuatorPositionValue)
        except TypeError:
            print "PositionUpdateErrorGI"  # Se prévenir des erreur de lecture quand la readline fait de la merde
        self.recursive_func_log.append(self.root.after(100, self.ActuatorPositionUpdate))

    def LoadTareAction(self):
        self.LoadTare = self.LoadValue
        return self.LoadTare

    def LoadUpdate(self):
      try:
	self.LoadValue = self.LoadValue-self.LoadTare
	self.LoadValue = round(self.LoadValue,5)
	self.LoadLabel.configure(text=self.LoadValue)
      except TypeError:
	print "LoadUpdateErrorGI"
      self.recursive_func_log.append(self.root.after(10,self.LoadUpdate))

    def change_button_level(self):
        self.pid_on = not self.pid_on
        if self.pid_on:
	  #self.PIDTitle.grid_forget()
	  self.PIDButtonStart.grid_forget()
	  self.PIDButtonStop.grid_forget()
	  self.ActuatorSpeedPID.grid_forget()
	  self.ActuatorSpeedPIDSpinbox.grid_forget()
	  self.LoadPID.grid_forget()
	  self.LoadPIDSpinbox.grid_forget()
	  
	else:
	  #self.PIDTitle.grid(row=1,column=1,sticky="w",padx=10,pady=10,in_=self.framePID)
	  self.PIDButtonStart.grid(row=2,column=1,sticky="w",padx=10,pady=10,in_=self.framePID)
	  self.PIDButtonStop.grid(row=3,column=1,sticky="w",padx=10,pady=10)
          self.ActuatorSpeedPID.grid(row=2, column=2, sticky= "w",padx=10,pady=10,in_=self.framePID)
          self.ActuatorSpeedPIDSpinbox.grid(row=2, column=3, sticky= "w",padx=10,pady=10)
          self.LoadPID.grid(row=3, column=2, sticky= "w",padx=10,pady=10)
          self.LoadPIDSpinbox.grid(row=3, column=3, sticky= "w",padx=10,pady=10)
	  

    def go(self):
        if self.flag == 0:
            self.flag = 1
            # self.RecordDataNumberLabel.configure(text=self.cpt)
        # self.t0 = time.time() #Initialisation du temps
        # self.StartRecordData()
        print "RecordDataStart"

    def stopRecord(self):
        self.flag = 0
        print "RecordDataStop"

    def PIDgo(self):
	if self.PIDflag ==0:    
	  self.PIDflag=1
	#self.RecordDataNumberLabel.configure(text=self.cpt)
	  #self.t0 = time.time() #Initiaself.ActuatorSpeedPIDlisation du temps
	  #self.StartRecordData()
	print "PIDactivated"
	
    def PIDstop(self):
        self.PIDflag = 0
        time.sleep(0.01)
        self.technical.set_speed(0) #prevent actuator to move once the button PIDStop is hit. But, need to be hit twice if time.sleep is not here. Amazing shity code.
        print "PIDdeactivated"

        # def recordflag(self): #creer une fonction qui envoie la valeur de flag
        # self.recordflag_pipe_send.send(self.flag)
        # print flag

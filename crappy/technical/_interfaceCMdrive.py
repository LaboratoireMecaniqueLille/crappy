# -*- coding:utf-8 -*-
import Tix
from Tkinter import *
#from serial.tools import list_ports
import serial
from crappy.actuator import CmDrive

class Interface(Frame):
    """Creat a graphic interface that permit to connect the motor via a serial port, and to send command in terms of speed or position"""
    def __init__(self, root, **kwargs):
	
	self.videoInstron = CmDrive()
	Frame.__init__(self, root, width=1000, height=1000, **kwargs)
	root.geometry("395x230")
	root.title("Pilotage")
	
	#Création des onglets
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
	#fin création des onglets
	
	# Début onglet 1
	sous_fra = Frame(fra1, width=400, borderwidth=2,relief=GROOVE) # create a frame in canvas(p1)
	self.portLabel = Label(sous_fra, text="Serial port:") # create a label 
        self.myPortCombo = Tix.StringVar() # create a variable, it will contain port selection 
	#self.portCombo = Tix.ComboBox(sous_fra, editable=1, dropdown=1, variable=self.myPortCombo) # create a combobox, it will contain names of ports 
	#self.portCombo.entry.config(state='readonly') # configure the combobox in read only
	#self.ma_liste= ['/dev/ttyS0'] # create a list
        #self.ma_liste=list_ports.comports() # add ports's names into the list
        #self.ma_liste=[elem[0] for elem in self.ma_liste] 
	#nbElement=len(self.ma_liste)
        #i=0
        #while(i<nbElement):
            #self.portCombo.insert(0, self.ma_liste[i]) # add the list into the combobox
            #i=i+1
            
	self.portCombo= Entry(sous_fra,textvariable= self.myPortCombo)
	self.baudrateLabel = Label(sous_fra, text="Baudrate:") # create a label
        self.baudCombo = Tix.StringVar() # create a variable, it will contain baudrate selection
	#self.baudrateCombo = Tix.ComboBox(sous_fra, editable=1, dropdown=1, variable=self.baudCombo) # create a combobox, it will contain baudrate values
	#self.baudrateCombo.entry.config(state='readonly') # configure the combobox in read only
        #baudList=[4800,9600,19200,38400,57600,115200] # create a list with differents values of baudrate
        #i=0
        #while(i<6):
	    #self.baudrateCombo.insert(END,baudList[i]) # add baudList into the combobox
	    #i=i+1  
	self.baudrateCombo = Entry(sous_fra, textvariable=self.baudCombo)
	self.connection = Button(sous_fra, text="Connect", command= self.connection) # create connect Button
	sous_fra1 = Frame(fra1, width=400) # create a second frame in canvas(p1)
	#self.enable = Button(sous_fra1, text="Enable Motor", command= self.enable_Motor, width=10) # create enable motor Button
	#self.enable.config(state='disabled') # disable the state of the button
	#self.disable= Button(sous_fra1, text="Disable Motor", command= self.disable_Motor, width=10) # create disable motor button
	#self.disable.config(state='disabled') # disable the state of he button
	#self.resetZero = Button(sous_fra1, text="Reset Servostar", command= self.reset_Servostar, width=10) # create reset servostar button
	#self.resetZero.config(state='disabled') # disable the state of the button
	self.location = Button(sous_fra1, text="Examine location", command= self.examineLocation, width=10) # create examine location button
	self.location.config(state='disabled') # disable the state of the button
	locationLabel = Label(sous_fra1, text="Location:") # create 'location:' label
	self.Resultat = StringVar() # create a variable, it will contain the result of examineLocation method
	self.Resultat.set("0") # set the variable to zero
	resultatLabel = Label(sous_fra1,textvariable=self.Resultat) # create a label, it will show the variable
	self.resetZero= Button(sous_fra1, text= " resetZero CMdrive", command = self.reset_servostar)
	self.quit= Button(sous_fra1, text="Cancel", command= root.quit) # create cancel button
	#Fin onglet 1
	
	# Début onglet 2
	sous_fra2= Frame(fra2, width=400) # create a frame in canvas(p2)
	sous_fra2bis= Frame(fra2, width=400) # create a second fram in canvas(p2)
	#self.init_modeButton = Button(sous_fra2, text="Initialize Steering Mode", command= lambda: self.init_steering_mode(1), width=15) # create Initialize steering Mode button 
	#self.init_modeButton.config(state='disabled') # disable the state of the button
	self.speedVar = StringVar() # creaate a variable, it will contain the value of speedEntry
	speedLabel = Label(sous_fra2bis, text="Velocity:") # create 'Velocity:' label
	speedEntry = Entry(sous_fra2bis, textvariable=self.speedVar, width=5) # create an entry, it will contain a velocity choose by the user
	self.advanceButton = Button(sous_fra2bis, text= "Go up", command=self.advance, width=10) # create a advance button
	self.advanceButton.config(state='disabled') # disable the state of the button
	self.recoilButton = Button(sous_fra2bis, text="Go down", command= self.recoil, width=10) # create recoil button
	self.recoilButton.config(state='disabled') # disable the state of the button
	stopVMButton = Button(sous_fra2bis, text="STOP", command= self.stopMotion, width=10) # create stop button
	defineZero = Button(sous_fra2bis, text="Define Zero", command= self.defineZero, width=10)
	#Début onglet 3
	sous_fra3_2 = Frame(fra3, width=400) # create a frame in canvas(p3)
	#self.init_mode = Button(sous_fra3_2, text="Initialize Steering Mode", command= lambda: self.init_steering_mode(2), width=15) # create Initialize Steering Mode button
	#self.init_mode.config(state='disabled') # disable the state of the button
	self.moveZeroButton = Button(sous_fra3_2, text="Move Home", command= self.moveZero, width=15) # create move home button
	self.moveZeroButton.config(state='disabled') # disable the state of the button
	#orderLabel = Label(sous_fra3_2, text="order :") # create 'order:' label
	#self.order= StringVar() # create a variable, it will contain the value of entry1
        #self.entry1 = Entry(sous_fra3_2, textvariable= self.order) # create an entry, it will contain the number of order choose by the user
        #self.entry1.focus_set() # pick out the widget that will receive keyboard events
        
        positionLabel = Label(sous_fra3_2, text="position :")# create 'position:' label
	self.position= StringVar()# create a variable, it will contain the value of entry2
        self.entry2 = Entry(sous_fra3_2, textvariable= self.position)# create an entry, it will contain the positon choose by the user
        self.entry2.focus_set()# pick out the widget that will receive keyboard events
        
        
        #velocityLabel = Label(sous_fra3_2, text="velocity :")# create 'velocity:' label
	#self.velocity= StringVar()# create a variable, it will contain the value of entry3
        #self.entry3 = Entry(sous_fra3_2, textvariable= self.velocity)# create an entry, it will contain the velocity choose by the user
        #self.entry3.focus_set()# pick out the widget that will receive keyboard events
        
        
        motionTypeLabel = Label(sous_fra3_2, text="motion type :")# create 'motionType:' label
	self.motionType= StringVar()# create a variable, it will contain the value of entry4
	#self.motionTypeCombo = Tix.ComboBox(sous_fra3_2, editable=1, dropdown=1,  variable=self.motionType, listwidth=30) # create a combobox, it will contain motion type values
	#self.motionTypeCombo.entry.config(width=10, state='readonly') # configure the combobox in read only
	#self.motionTypeCombo.insert(END,"relative")# add motion type into the combobox
	#self.motionTypeCombo.insert(END,"absolute")# add motion type into the combobox
	
	sous_fra3 = Frame(fra3, width=400) # create a frame in canvas(p3)
	#self.moveNumber = StringVar() # create a variable, it will contain the value of moveSpinbox
	#self.moveNumber.set(0) # set the variable to zero
	sous_sous_fra3= Frame(sous_fra3, width=400) # create a frame in sous_fra3
	## Création d'un widget Spinbox
	#moveSpinbox = Spinbox(sous_sous_fra3,from_=0,to=100,increment=1,textvariable=self.moveNumber,width=5) # create a spinbox widget
	self.moveButton = Button(sous_sous_fra3, text="Move", command= self.move) # create move button
	self.moveButton.config(state='disabled') # disable the state of the button
	stopMTButton = Button(sous_sous_fra3, text="STOP", command= self.stopMotion) # create STOP button
	
	sous_sous_fra3bis= Frame(sous_fra3, width=400) # create a second frame in sous_fra3
	#infoLabel = Label(sous_sous_fra3bis, text='Get informations about motion task n°:') # create a label
	#self.moveInfoNumber = StringVar() # create a variable, it will contain the value of infoSpinbox
	#self.moveInfoNumber.set(0) # set the varianle to zero
	#infoSpinbox = Spinbox(sous_sous_fra3bis, from_=0, to=100, increment=1, textvariable=self.moveInfoNumber, width=5) # create a spinbox widget
	#self.infoButton = Button(sous_sous_fra3bis, text="OK", command = self.get_informationMT) # create a info Button
	#self.infoButton.config(state='disabled')# disable the state of the button
	#fin onglet 3
	
		
	
	
	# placement des widgets onglet 1
	# show widgets on canvas(p1) 
	sous_fra.grid(padx=10,pady=10) # 
	self.portLabel.grid(row=1, column=0, sticky= "sw",padx=10,pady=10)
	self.portCombo.grid(row=1, column=1, sticky= "sw",padx=10,pady=10)
	self.baudrateLabel.grid(row=2, column=0, sticky= "sw",padx=10,pady=10)
	self.baudrateCombo.grid(row=2, column=1, sticky= "sw",padx=10,pady=10)
	self.connection.grid(row=3, column=1,sticky= "se")
	# placement widget sous frame onglet 1
	# show widgets on frame canvas(p1)
	sous_fra1.grid()
	#self.enable.grid(row=1, column=0,sticky= "sw", padx=10,pady=10)
	#self.disable.grid(row=2, column=0,sticky= "sw", padx=10,pady=10)
	#self.resetZero.grid(row=3, column=0,sticky= "sw", padx=10,pady=10)
	self.location.grid(row=4, column=0,sticky= "s", padx=10,pady=10)
	locationLabel.grid(row=4, column=1)
	resultatLabel.grid(row=4, column=2)
	self.resetZero.grid(row=4, column=4)
	#self.quit.grid(row=4, column=4, sticky= "e")
	
	#placement des widgets onglet 2
	# show widgets on canvas(p2)
	sous_fra2.grid(row=0, column=0, padx=10,pady=10)
	sous_fra2bis.grid(row=1, column=0, padx=10,pady=10)
	#self.init_modeButton.grid(row=0, column=0, padx=10,pady=10)
	speedLabel.grid(row=0, column=0, sticky='w')
	speedEntry.grid(row=0, column=2, sticky='w')
	self.recoilButton.grid(row=0, column=3, sticky='w')
	self.advanceButton.grid(row=1, column=3,sticky='w')
	stopVMButton.grid(row=2, column=3,sticky='w')
	defineZero.grid(row=3, column=3,sticky='w')
	
	# placement des widgets onglet 3
	#  show widgets on canvas(p3)
	sous_fra3_2.grid(padx=10,pady=10)
	self.moveZeroButton.grid(row=0, column=0, padx=10,pady=10)
	positionLabel.grid(row=1, column=0,sticky='w')
	motionTypeLabel.grid(row=2, column=0,sticky='w')
	self.entry2.grid(row=1, column=1,sticky='w')
	Radiobutton(sous_fra3_2, text="absolute", variable=self.motionType, value=True).grid()
	Radiobutton(sous_fra3_2, text="relative", variable=self.motionType, value=False).grid()
	sous_fra3.grid(row=3, column=0)
	sous_sous_fra3.grid(row=0, column=0)
	sous_sous_fra3bis.grid(row=1, column=0)
	self.moveButton.grid(row=0, column=1)
	stopMTButton.grid(row=0, column=2)
	#show notebooks
	monnotebook.pack(side=LEFT, fill=Tix.BOTH, expand=1, padx=5, pady=5)
	
    # function to initialize the connection 	    
    def connection(self):
      if self.myPortCombo.get() == "" or self.baudCombo.get() == "":
	print 'you must choose port configuration'
      else:
	try:
	    self.ser = self.videoInstron.setConnection(self.myPortCombo.get(), self.baudCombo.get())
#	    self.vm = videoInstron(self.ser)
	    print 'connection'
	    self.location.config(state='normal')
	    self.moveButton.config(state='normal')
	    self.recoilButton.config(state='normal')
	    self.advanceButton.config(state='normal')
	    self.moveZeroButton.config(state='normal')
	    
	except:
	   print ' Connection error'
	
    def defineZero(self):
      self.ser.close()
      self.ser.open()
      self.ser.write('P=0\r')
      self.ser.readline()
      self.ser.close()
      
    def reset_servostar(self):
      self.ser.close()
      self.ser.open()
      self.ser.write('FD\r')
      self.ser.readline()
      self.ser.close()
      
    # function to examine the location of the motor
    def examineLocation(self):
      location=self.videoInstron.examineLocation(self.ser)
      self.Resultat.set(location)
    
    # function to move home
    def moveZero(self):
      self.videoInstron.moveZero()
      print 'moving home'
    
    # function to apply a motion task 
    def move(self):
      if self.motionType.get()== "" or self.position.get()=="":
	print "one of the entry is empty"  
      else:
	if self.motionType.get() == '1':
	    self.videoInstron.applyAbsoluteMotion(int(self.position.get()))
	    print 'MA mode'
	else:
	    self.videoInstron.applyRelativeMotion(int(self.position.get()))
	    print 'MR mode'
    
      
    # function to advance the motor on velocity mode  
    def advance(self):
      if self.speedVar.get() == "":
	print 'choose velocity'
      else:
	self.videoInstron.applyPositiveSpeed(int(self.speedVar.get()))
	print('the motor go up with speed=%i' % int(self.speedVar.get()))
    
    #function to recoil the motor on velocity mode
    def recoil(self):
      if self.speedVar.get() == "":
	print 'choose velocity'
      else:
	self.videoInstron.applyNegativeSpeed(int(self.speedVar.get()))
	print('the motor go down with speed=%i' % int(self.speedVar.get()))
    
    # function to stop a motion
    def stopMotion(self):
      self.videoInstron.stopMotion()
      print ' Motion has been stoped'
    
if __name__ == '__main__':
	root = Tix.Tk()
	interface = Interface(root)
	interface.mainloop()
	interface.destroy()

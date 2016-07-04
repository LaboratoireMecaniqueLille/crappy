import crappy2 as crappy 
import numpy as np
import time
#from interface import *
import Tix
from Tkinter import *
import tkFont
import tkFileDialog

crappy.blocks.MasterBlock.instances=[]

t0=time.time()

class conditionfiltree(crappy.links.MetaCondition):
	def __init__(self,labels=[],mode="mean",size=10):
		self.mode=mode
		self.size=size
		self.labels=labels
		self.FIFO=[[] for label in self.labels]
		self.test=False
		self.blocking=False
	
	def evaluate(self,value):
	#print "1"
		recv=self.external_trigger.recv(blocking=self.blocking) # first run is blocking, others are not
		self.blocking=False
		if recv == 1:
			print 'EUREKA'
			self.test=True
			
		elif recv == 0:
			print 'je recois rien'
			self.test=False
		
		for i,label in enumerate(self.labels):
			#print self.FIFO[i]
			self.FIFO[i].insert(0,value[label])
			if len(self.FIFO[i])>self.size:
				self.FIFO[i].pop()
			if self.mode=="median":
				result=np.median(self.FIFO[i])
			elif self.mode=="mean":
				result=np.mean(self.FIFO[i])
			value[label+"_filtered"]=result

		
		if self.test:
			return value
		else:
			return None


try:
	#creating objects
	comediSensor=crappy.sensor.ComediSensor(channels=[0],gain=[20613],offset=[0])
	t,F0=comediSensor.get_data(0)
	print "offset effort=", F0
	comediSensor=crappy.sensor.ComediSensor(channels=[1],gain=[4125],offset=[0]) #5100
	t,V0=comediSensor.get_data(0)
	print "offset vitesse=", V0
	comediSensor=crappy.sensor.ComediSensor(channels=[2],gain=[500],offset=[0])
	t,C0=comediSensor.get_data(0)
	print "offset couple=", C0
	
	comediSensor=crappy.sensor.ComediSensor(channels=[0,1,2],gain=[20613,4125,-500],offset=[-F0,-V0,C0])
	
	effort=crappy.blocks.MeasureByStep(comediSensor,labels=['t(s)','F(N)','Vit','Couple'],freq=500)
	VariateurTribo=crappy.technical.VariateurTribo(port='/dev/ttyS4')#,port_arduino='/dev/ttyACM0')
	labjack = crappy.actuator.LabJackActuator(channel = "TDAC0", gain = 1./399.32, offset = -17.73/399.32)
	labjack.set_cmd(0)
	labjack.set_cmd_ram(0,46002) #sets the pid off
        labjack.set_cmd_ram(0,46000) #sets the setpoint at 0 newton
	#comediOut=crappy.actuator.ComediActuator(device='/dev/comedi0',subdevice=1,channel=0,range_num=0,gain=1.0/1600.0,offset=0)
	#comediOut.set_cmd(0)
	#VariateurTribo.actuator.go_effort(0)
	saver=crappy.blocks.Saver("/home/tribo/save_dir/openlog.txt")
	  
	compacter=crappy.blocks.Compacter(100)

	graph=crappy.blocks.Grapher("dynamic",('t(s)','F(N)'))
	graph2=crappy.blocks.Grapher("dynamic",('t(s)','Vit'))
	graph3=crappy.blocks.Grapher("dynamic",('t(s)','Couple'))
	#graph4=crappy.blocks.Grapher("dynamic",('t(s)','signal'))
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link6=crappy.links.Link()
	link5=crappy.links.Link(condition=conditionfiltree())
	link5.add_external_trigger(link6)
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	
	#links
	effort.add_output(link1)
	effort.add_output(link5)
	compacter.add_input(link1)
	compacter.add_output(link2)
	compacter.add_output(link3)
	compacter.add_output(link4)
	compacter.add_output(link7)
	#compacter.add_output(link8)
	graph.add_input(link2)
	graph2.add_input(link3)
	graph3.add_input(link4)
	#graph4.add_input(link8)
	saver.add_input(link7)

	

	


	
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.t0 = t0
		
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.start()
	root = Tix.Tk()
	interface=crappy.blocks.InterfaceTribo(root,VariateurTribo,labjack)#,link5,link6)
	interface.root.protocol("WM_DELETE_WINDOW", interface.on_closing)
	interface.add_input(link5)
	interface.add_output(link6)
	print 'top1'
	interface.mainloop()	

	
	try:
		var = interface.getInfo()
		root.destroy()
		print var
	except Exception as e:
		print "Error: ", e
		sys.exit(0)
	#GI_Cegitab(VariateurTribo,comediOut)
	
except KeyboardInterrupt:
	VariateurTribo.actuator.stop_motor()
	labjack.set_cmd(0)
	labjack.set_cmd_ram(-41,46000)
	labjack.set_cmd_ram(0,46002)
	time.sleep(1)
	labjack.close()
	time.sleep(0.1)
	VariateurTribo.close()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.stop()
except Exception as e:
	print e
finally:
	#labjack.set_cmd(0)
	#labjack.set_cmd_ram(0,46000)
	#labjack.set_cmd_ram(0,46002)
	time.sleep(0.1)
	VariateurTribo.close()
	print "Hasta la vista Baby"

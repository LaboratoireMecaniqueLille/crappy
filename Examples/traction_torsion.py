import time
import numpy as np
import crappy
crappy.blocks.MasterBlock.instances=[] # Init masterblock instances


t0=time.time()
try:
########################################### Creating objects
	
	instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2,3],gain=[50,10000,45,1000]) # 10 times the gain on the machine if you go through an usb dux sigma
	cmd_traction=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=1, range_num=0, gain=1, offset=0)
	cmd_torsion=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=1, offset=0)

########################################### Initialising the outputs

	cmd_torsion.set_cmd(0)
	cmd_traction.set_cmd(0)
	
########################################### Creating blocks
	
	stream=crappy.blocks.MeasureComediByStep(t0,instronSensor,labels=['t(s)','dep(mm)','f(N)','angle(deg)','C(Nm)'])
	traction=crappy.blocks.PathGenerator(t0, send_freq=400, actuator=cmd_traction, waveform=['sinus','sinus','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[1,2,4], offset=[0,0,0], phase=[0,0,0], repeat=True)
	torsion=crappy.blocks.PathGenerator(t0, send_freq=400, actuator=cmd_torsion, waveform=['sinus','triangle','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[0,0,0], offset=[0,0,0], phase=[np.pi,np.pi,np.pi], repeat=True)
	compacter=crappy.blocks.Compacter(100)
	save=crappy.blocks.Saver("/home/essais-2015-1/Bureau/t_dep_F_angle_C.txt")
	graph=crappy.blocks.Grapher("dynamic",(0,1))
	graph_stat=crappy.blocks.Grapher("dynamic",(0,2))
	graph2=crappy.blocks.Grapher("dynamic",(0,3))
	graph3=crappy.blocks.Grapher("dynamic",(0,4))
	
########################################### Creating links
	
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	
########################################### Linking objects
	stream.add_output(link1)
	
	compacter.add_input(link1)
	
	compacter.add_output(link2)
	compacter.add_output(link3)
	compacter.add_output(link5)
	compacter.add_output(link6)
	compacter.add_output(link7)
	
	save.add_input(link2)
	
	graph.add_input(link3)
	
	graph_stat.add_input(link5)
	graph2.add_input(link6)
	graph3.add_input(link7)
########################################### Starting objects

	for instance in crappy.blocks.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution


########################################### Stopping objects

	#for instance in crappy.blocks.MasterBlock.instances:
		#instance.stop()

except KeyboardInterrupt:
	for instance in crappy.blocks.MasterBlock.instances:
		instance.stop()
import time
import numpy as np
import crappy2
crappy2.blocks._masterblock.MasterBlock.instances=[] # Init masterblock instances


t0=time.time()
try:
########################################### Creating objects
	instronSensor=crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3, 4, 5], gain=[1, 1, 1, 1, 1, 1])
	#instronSensor=crappy2.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10000]) # 10 times the gain on the machine if you go through an usb dux sigma
	#cmd_traction=crappy2.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=1, range_num=0, gain=1, offset=0)
	#cmd_torsion=crappy2.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=1, offset=0)

########################################### Initialising the outputs

	#cmd_torsion.set_cmd(0)
	#cmd_traction.set_cmd(0)
	
########################################### Creating blocks
	#send_freq=400, actuator=cmd_traction, waveform=['sinus','sinus','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[1,2,4], offset=[0,0,0], phase=[0,0,0], repeat=True
	#send_freq=400, actuator=cmd_torsion, waveform=['sinus','triangle','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[0,0,0], offset=[0,0,0], phase=[np.pi,np.pi,np.pi], repeat=True
	#stream=crappy2.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','dep(mm)','f(N)'],freq=100)
	stream=crappy2.blocks.StreamerComedi(instronSensor, labels=['t(s)', 'def(%)', 'F(N)', 'dist(deg)', 'C(Nm)', 'dep(mm)', 'ang(deg)'], freq=2000.)
	#traction=crappy2.blocks.SignalGenerator(path=[{"waveform":"sinus","time":100,"phase":0,"amplitude":2,"offset":0.5,"freq":2.5}],
												#send_freq=400,repeat=True,labels=['t(s)','signal'])
	#torsion=crappy2.blocks.SignalGenerator(path=[{"waveform":"triangle","time":50,"phase":0,"amplitude":5,"offset":-0.5,"freq":1}]
											#,send_freq=400,repeat=False,labels=['t(s)','signal'])
	compacter=crappy2.blocks.Compacter(5)
	#compacter2=crappy2.blocks.Compacter(400)
	save=crappy2.blocks.Saver("/home/corentin/Bureau/streamer_delete.txt")
	graph=crappy2.blocks.Grapher("dynamic", ('ang(deg)', 'dep(mm)'))
	#graph_stat=crappy2.blocks.Grapher("dynamic",(0,2))
	graph2=crappy2.blocks.Grapher("dynamic", ('t(s)', 'ang(deg)'), ('t(s)', 'dep(mm)'))
	#graph3=crappy2.blocks.Grapher("dynamic",(0,4))
	
########################################### Creating links
	
	link1=crappy2.links.Link(crappy2.links.Filter(labels=['dist(deg)'], mode="median", size=50))
	link2=crappy2.links.Link()
	link3=crappy2.links.Link()
	link4=crappy2.links.Link()
	link5=crappy2.links.Link()
	link6=crappy2.links.Link()
	link7=crappy2.links.Link()
	
########################################### Linking objects
	stream.add_output(link2)
	#traction.add_input(link5)
	#torsion.add_output(link2)
	#torsion.add_output(link5)
	
	compacter.add_input(link2)
	#compacter2.add_input(link2)
	
	compacter.add_output(link3)
	compacter.add_output(link4)
	compacter.add_output(link5)
	#compacter.add_output(link6)
	#compacter.add_output(link7)
	
	save.add_input(link5)
	
	graph.add_input(link3)
	
	#graph_stat.add_input(link5)
	graph2.add_input(link4)
	#graph3.add_input(link7)
########################################### Starting objects
	t0=time.time()
	for instance in crappy2.blocks._masterblock.MasterBlock.instances:
		instance.t0(t0)
		
	for instance in crappy2.blocks._masterblock.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution


########################################### Stopping objects

	#for instance in crappy2.blocks.MasterBlock.instances:
		#instance.stop()

except KeyboardInterrupt:
	for instance in crappy2.blocks._masterblock.MasterBlock.instances:
		instance.stop()
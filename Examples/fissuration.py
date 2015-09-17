import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances


t0=time.time()

try:
########################################### Creating objects
	
	#instronSensor=crappy.sensor.ComediSensor(channels=[0],gain=[-48.8],offset=[0])
	#t,F0=instronSensor.getData(0)
	#instronSensor=crappy.sensor.ComediSensor(channels=[0],gain=[-48.8],offset=[-F0])
	#biotensTech=crappy.technical.Biotens(port='/dev/ttyUSB0', size=15)

########################################### Creating blocks
	
	#compacter_coeff=crappy.blocks.Compacter(400)
	#graph_coeff=crappy.blocks.Grapher("dynamic",('t(s)','coeff'))
	
	compacter_signal=crappy.blocks.Compacter(200)
	graph_signal=crappy.blocks.Grapher("dynamic",('t(s)','signal'))


	coeffGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":8000,"freq":0.02},
														{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":7000,"freq":0.02},
														{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":2000,"freq":0.02}],
							send_freq=100,repeat=True,labels=['t(s)','coeff'])
	
	signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"sinus","time":100,"phase":0,"amplitude":0.45,"offset":0.55,"freq":2}],
							send_freq=800,repeat=True,labels=['t(s)','signal'])
	
	
	adapter=crappy.blocks.SignalAdapter(initial_coeff=0,delay=5,send_freq=800,labels=['t(s)','signal'])
	compacter_adapter=crappy.blocks.Compacter(400)
	graph_adapter=crappy.blocks.Grapher("dynamic",('t(s)','signal'))


########################################### Creating links
	
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	#link9=crappy.links.Link()
	#link10=crappy.links.Link()
	#link11=crappy.links.Link()
	
########################################### Linking objects

	signalGenerator.add_output(link4)
	signalGenerator.add_output(link7)
	compacter_signal.add_input(link7)
	compacter_signal.add_output(link8)
	graph_signal.add_input(link8)
	
	coeffGenerator.add_output(link1)
	#coeffGenerator.add_output(link2)
	#compacter_coeff.add_input(link2)
	#compacter_coeff.add_output(link3)
	#graph_coeff.add_input(link3)
	
	adapter.add_input(link1)
	adapter.add_input(link4)
	adapter.add_output(link5)
	compacter_adapter.add_input(link5)
	compacter_adapter.add_output(link6)
	graph_adapter.add_input(link6)

########################################### Starting objects

	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)

	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution


########################################### Stopping objects

except (Exception,KeyboardInterrupt) as e:
	print "Exception in main :", e
	#for instance in crappy.blocks._meta.MasterBlock.instances:
		#instance.join()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		try:
			instance.stop()
			print "instance stopped : ", instance
		except:
			pass
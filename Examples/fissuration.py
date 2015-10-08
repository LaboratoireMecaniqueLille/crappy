import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
import pandas as pd
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances
#for tracking memory leaks:
#from pympler import tracker
#tr = tracker.SummaryTracker()
#from pympler import summary
#from pympler import muppy
#sum1 = summary.summarize(all_objects)
#summary.print_(sum1) 
#sum2 = summary.summarize(muppy.get_objects())
#diff = summary.get_diff(sum1, sum2)
#summary.print_(diff)     

class condition_coeff(crappy.links.MetaCondition):
	def __init__(self):
		initial_coeff=0
		self.last_cycle=-1
		self.coeff=initial_coeff
		self.last_coeff=initial_coeff
		self.delay=5
		self.blocking=True
		
	def evaluate(self,value):
		recv=self.external_trigger.recv(blocking=self.blocking)
		self.blocking=False
		try:
			self.new_coeff=recv['coeff'][0]
			#print "new_coeff :", self.new_coeff
		except TypeError:
			pass
		if self.new_coeff!=self.coeff: # if coeff is changing
			if self.coeff==self.last_coeff: # if first change
				self.t_init=time.time()
				self.t1=self.t_init
			self.t2=time.time()
			if (self.t2-self.t_init)<self.delay:
				self.coeff+=(self.new_coeff-self.last_coeff)*((self.t2-self.t1)/(self.delay))
			else: # if delay is passed
				self.coeff=self.new_coeff
				self.last_coeff=self.coeff
			self.t1=self.t2
		print "coeff :", self.coeff
		value['signal'][0]*=self.coeff
		return value

class condition_cycle_bool(crappy.links.MetaCondition):
	def __init__(self):
		self.cycle=0
		self.last_cycle=-1
		
	def evaluate(self,value):
		cycle=value['cycle'][0]
		#print cycle
		#print cycle
		if cycle!=self.last_cycle:
			self.last_cycle=cycle
			#self.go=False
			return value
		else:
			return None

class condition_K(crappy.links.MetaCondition):
	def __init__(self):
		self.K=0
		#self.last_cycle=-1
		
	def evaluate(self,value):
		self.K=0*(value['tension'][0])+0*(1.18/1.63)+0.2
		value['coeff'] = pd.Series((self.K), index=value.index)
		#print value
		return value

try:
########################################### Creating objects
	
	instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0],gain=[10],offset=[0])
	#agilentSensor=crappy.sensor.Agilent34420ASensor(device='/dev/ttyUSB0',baud_rate=9600,timeout=1)
	agilentSensor=crappy.sensor.DummySensor()
	comedi_actuator=crappy.actuator.ComediActuator(device='/dev/comedi1',subdevice=1,channel=1,range_num=0,gain=1,offset=0)
	comedi_actuator.set_cmd(0)
	
########################################### Creating blocks
	tension=crappy.blocks.MeasureAgilent34420A(agilentSensor,labels=['t_agilent(s)','tension'])
	camera=crappy.blocks.StreamerCamera("Ximea",freq=None,save=True,save_directory="/home/corentin/Bureau/images_fissuration/")
	
	compacter_tension=crappy.blocks.Compacter(5)
	graph_tension=crappy.blocks.Grapher("dynamic",('t(s)','t_agilent(s)'),('t(s)','coeff'))
	
	effort=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','dep(mm)'],freq=400)
	compacter_effort=crappy.blocks.Compacter(400)
	graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','dep(mm)'))
	
	compacter_signal=crappy.blocks.Compacter(500)
	save_signal=crappy.blocks.Saver("/home/corentin/Bureau/signal.txt")
	#graph_signal=crappy.blocks.Grapher("dynamic",('t(s)','cycle'))


	#coeffGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":8000,"freq":0.02},
														#{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":7000,"freq":0.02},
														#{"waveform":"triangle","time":10,"phase":0,"amplitude":0,"offset":2000,"freq":0.02}],
							#send_freq=100,repeat=True,labels=['t(s)','coeff','cycle'])
	
	signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"sinus","time":1000000,"phase":0,"amplitude":0.45,"offset":0.55,"freq":1}],
							send_freq=500,repeat=True,labels=['t(s)','signal','cycle'])
	
	####CommandComedi([comedi_actuator])
	
	adapter=crappy.blocks.SignalAdapter(initial_coeff=0,delay=10,send_freq=600,labels=['t(s)','signal'])
	compacter_adapter=crappy.blocks.Compacter(500)
	graph_adapter=crappy.blocks.Grapher("dynamic",('t(s)','signal'))
	save_adapter=crappy.blocks.Saver("/home/corentin/Bureau/signal_adapted.txt")
	comedi_output=crappy.blocks.CommandComedi([comedi_actuator])


########################################### Creating links
	
	link1=crappy.links.Link(condition=condition_cycle_bool())
	link2=crappy.links.Link(condition=condition_cycle_bool())
	link3=crappy.links.Link(condition=condition_K())
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link(condition=condition_K())
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	link9=crappy.links.Link()
	link10=crappy.links.Link()
	link11=crappy.links.Link()
	link12=crappy.links.Link()
	link13=crappy.links.Link()
	link14=crappy.links.Link()
	link15=crappy.links.Link()
	link16=crappy.links.Link()
	
########################################### Linking objects

	camera.add_input(link1)
	
	tension.add_input(link2)
	tension.add_output(link3)
	tension.add_output(link6)
	
	compacter_tension.add_input(link3)
	compacter_tension.add_output(link5)
	
	graph_tension.add_input(link5)
	
	effort.add_output(link9)
	compacter_effort.add_input(link9)
	compacter_effort.add_output(link10)
	
	graph_effort.add_input(link10)
	
	signalGenerator.add_output(link1)
	signalGenerator.add_output(link2)
	signalGenerator.add_output(link7)
	signalGenerator.add_output(link13)
	
	adapter.add_input(link6)
	adapter.add_input(link7)
	adapter.add_output(link8)
	adapter.add_output(link15)
	
	compacter_signal.add_input(link13)
	compacter_signal.add_output(link14)
	save_signal.add_input(link14)
	#graph_signal.add_input(link14)
	
	#coeffGenerator.add_output(link1)
	#coeffGenerator.add_output(link2)
	#compacter_coeff.add_input(link2)
	#compacter_coeff.add_output(link3)
	#graph_coeff.add_input(link3)
	
	#adapter.add_input(link1)
	#adapter.add_input(link4)
	#adapter.add_output(link5)
	comedi_output.add_input(link8)
	compacter_adapter.add_input(link15)
	compacter_adapter.add_output(link12)
	compacter_adapter.add_output(link16)
	save_adapter.add_input(link16)
	graph_adapter.add_input(link12)

########################################### Starting objects

	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)

	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution
	#time.sleep(10)
	#sum1 = summary.summarize(muppy.get_objects())
	#summary.print_(sum1)
	#while True:
		#time.sleep(10)
		#sum2 = summary.summarize(muppy.get_objects())
		#diff = summary.get_diff(sum1, sum2)
		#summary.print_(diff)
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
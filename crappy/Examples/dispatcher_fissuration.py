#from multiprocessing import Pipe,Process
import time
import crappy
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances

class condition_cycle(crappy.links.Condition):
	def __init__(self):
		self.cycle=0
		self.go=True
		
	def evaluate(self,value):
		if value[2]>=F_max and self.go==True:
			self.cycle+=1
			self.go=False
		if value[2]<=F_min and self.go==False:
			self.go=True
		return self.cycle

class condition_cycle_bool(crappy.links.Condition):
	def __init__(self,n=1):
		self.cycle=0
		self.go=True
		self.n=n
		
	def evaluate(self,value):
		if value[2]>=F_max and self.go==True:
			self.cycle+=1
			self.go=False
			if self.cycle%self.n==0:
				return True
			else:
				return False
		elif value[2]<=F_min and self.go==False:
			self.go=True
			if self.cycle%self.n==0:
				return True
			else:
				return False
		else:
			return False



#############################################################################



#class condition_cycle_old(object):
	#def __init__(self):
		#self.cycle=0
		#self.go=True
		
	#def evaluate(self,value):
		#if value[2]>=F_max and self.go==True:
			#self.cycle+=1
			#self.go=False
		#if value[2]<=F_min and self.go==False:
			#self.go=True
		#bool_condition=True
		#ret_value=self.cycle
		#return (bool_condition,ret_value)

#class condition_cycle_bool_old(object):
	#def __init__(self,n=1):
		#self.cycle=0
		#self.go=True
		#self.n=n
		
	#def evaluate(self,value):
		##print value
		#if value[2]>=F_max and self.go==True:
			#self.cycle+=1
			#self.go=False
			#if self.cycle%self.n==0:
				##print "frame!"
				#return (True,True)
			#else:
				#return (False,False)
		#elif value[2]<=F_min and self.go==False:
			#self.go=True
			#if self.cycle%self.n==0:
				#return (True,True)
			#else:
				#return (False,False)
		#else:
			#return (False, False)

F_max=3.1e-5
F_min=2.9e-5
#t0=time.time()
try:
########################################### Creating objects
	
	instronSensor=crappy.sensor.ComediSensor(channels=[0,1],gain=[1,1])
	#cameraSensor=crappy.technical.Ximea()
	#agilentSensor=crappy.sensor.Agilent34420ASensor()

########################################### Creating blocks
	
	stream=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','dep(mm)','F(N)'],freq=200)
	camera=crappy.blocks.StreamerCamera("Ximea",freq=None,save=True,save_directory="./images_fissuration/")
	#resistance=crappy.blocks.MeasureAgilent34420A(agilentSensor)
	compacter=crappy.blocks.Compacter(100)
	compacter_resistance=crappy.blocks.Compacter(10)
	save=crappy.blocks.Saver("/home/corentin/Bureau/t_F_dep_cycle.txt")
	save_resistance=crappy.blocks.Saver("/home/corentin/Bureau/t_Res_cycle.txt")
	graph=crappy.blocks.Grapher("dynamic",('t(s)','F(N)'))
	graph_resistance=crappy.blocks.Grapher("dynamic",('t(s)','R(Ohm)'))
	
########################################### Creating links
	
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link(condition=condition_cycle_bool())
	link5=crappy.links.Link(condition=condition_cycle_bool(n=5))
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	link9=crappy.links.Link(condition=condition_cycle())
	link10=crappy.links.Link(condition=condition_cycle())
	
########################################### Linking objects
	stream.add_output(link1)
	#stream.add_output(link4)
	stream.add_output(link5)
	stream.add_output(link9)
	stream.add_output(link10)
	
	camera.add_input(link5)
	#resistance.add_input(link4)
	#resistance.add_output(link6)
	compacter.add_input(link1)
	compacter.add_input(link9)
	compacter.add_output(link2)
	compacter.add_output(link3)
	
	compacter_resistance.add_input(link10)
	#compacter_resistance.add_input(link6)
	#compacter_resistance.add_output(link7)
	compacter_resistance.add_output(link8)
	
	save.add_input(link3)
	save_resistance.add_input(link8)
	graph.add_input(link2)
	#graph_resistance.add_input(link7)
	
########################################### Starting objects


	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)
		
	for instance in crappy.blocks.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution
	#time.sleep(1)
	#time.sleep(100)

########################################### Stopping objects

	#for instance in crappy.blocks.MasterBlock.instances:
		#instance.stop()

except (Exception,KeyboardInterrupt) as e:
	print "Exception in main :", e
	for instance in crappy.blocks._meta.MasterBlock.instances:
		try:
			instance.stop()
		except:
			pass
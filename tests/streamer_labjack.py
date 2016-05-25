import time
import numpy as np
import crappy
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances

#class condition_sub(crappy.links.MetaCondition):
	#def __init__(self):
		#pass
	
	#def evaluate(self,value):
		#calc=value['T']-(value['T3']+value['T2'])/2.
		#value.update({'T_mean':calc})
		#return value
	
#class condition_calib(crappy.links.MetaCondition):
	#def __init__(self):
		#self.coeff_T= [-1.14944097e+19,-6.22927505e+15,9.45905908e+12,3.20160146e+09,-1.81259800e+06,1.91565284e+04,2.77297727e+01]
		#self.coeff_T2 = [-5.37671145e+18,-1.08133635e+16,6.54277247e+12,4.91194020e+09,-1.47818544e+06,1.91198815e+04,2.77679671e+01]
		#self.coeff_T3 = [-7.70172105e+18,-8.61309675e+15,6.53446611e+12,4.30990401e+09,-1.34687552e+06,1.90248560e+04,2.80630626e+01]
		#self.T_init_FIFO=[]
		##self.T_filtered_init_FIFO=[]
		#self.T2_init_FIFO=[]
		#self.T3_init_FIFO=[]
		#self.i=0
	#def evaluate(self,value):
		#T=np.polyval(self.coeff_T2,value.pop('T'))
		#T2=np.polyval(self.coeff_T2,value.pop('T2'))
		#T3=np.polyval(self.coeff_T3,value.pop('T3'))
		##try:
			##T_filtered=np.polyval(self.coeff_T,value.pop('T_filtered'))
		##except:
			##pass
		#if self.i<100:
			#self.T_init_FIFO.append(T)
			#self.T2_init_FIFO.append(T2)
			#self.T3_init_FIFO.append(T3)
			##self.T_filtered_init_FIFO.append(T_filtered)
			#self.T_init=np.mean(self.T_init_FIFO)
			#self.T2_init=np.mean(self.T2_init_FIFO)
			#self.T3_init=np.mean(self.T3_init_FIFO)
			##self.T_filtered_init=np.mean(self.T_filtered_init_FIFO)
			#self.i+=1
		#calc=(T-self.T_init)-((T3-self.T3_init)+(T2-self.T2_init))/2.
		#value.update({'T_mean':calc,'T':T-self.T_init,'T2':T2-self.T2_init,'T3':T3-self.T3_init})
		#return value


#class eval_stress(crappy.links.MetaCondition):
	#def __init__(self):
		#self.surface=110.74*10**(-6)
		#self.I=np.pi*((25*10**-3)**4-(22*10**-3)**4)/32
		#self.rmoy=((25+22.)*10**(-3))/2
		#self.size=20
		#self.labels=['C(Nm)']
		#self.FIFO=[[] for label in self.labels]
		
	#def evaluate(self,value):
		##if self.k==1:
			##print value
		#for i,label in enumerate(self.labels):
			##print self.FIFO[i]
			#self.FIFO[i].insert(0,value[label])
			#if len(self.FIFO[i])>self.size:
				#self.FIFO[i].pop()
			#result=np.mean(self.FIFO[i])
			#value[label]=result
		#value['tau(Pa)']=((value['C(Nm)']/self.I)*self.rmoy)
		##value['sigma(Pa)']=(value['F(N)']/self.surface)
		##if self.k==1:
			##print value
		#return value
	
	
#t0=time.time()
try:
########################################### Creating objects
	#instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10])
	#sensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2,3],gain=[0.02,100000,0.01*2.,500]) # dist is multiplied by 2 to be correct
	#sensor_effort=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,500])
	#sensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2],gain=[1,1,1],offset=[0,0,0]) #262*10**-6,180*10**-6,169*10**-6
	#t,T=sensor.getData(0)
	#t,T1=sensor.getData(1)
	#t,T2=sensor.getData(2)
	#sensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2],gain=[1,1,1],offset=[-T,-T1,-T2]) 
	sensor=crappy.sensor.LabJackSensor(channels=[0,1,2],gain=[1,1,1],chan_range=10,mode="streamer",scanRate=4000,scansPerRead=1) # dist is multiplied by 2 to be correct
	#instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10000]) # 10 times the gain on the machine if you go through an usb dux sigma
	#cmd_traction=crappy.actuator.LabJackActuator(channel="TDAC2", gain=1, offset=0)
	#cmd_traction2=crappy.actuator.LabJackActuator(channel="TDAC3", gain=1, offset=0)
	#cmd_torsion=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=1, offset=0)

########################################### Initialising the outputs

	#cmd_torsion.set_cmd(0)
	#cmd_traction.set_cmd(0)
	
########################################### Creating blocks
	#send_freq=400, actuator=cmd_traction, waveform=['sinus','sinus','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[1,2,4], offset=[0,0,0], phase=[0,0,0], repeat=True
	#send_freq=400, actuator=cmd_torsion, waveform=['sinus','triangle','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[0,0,0], offset=[0,0,0], phase=[np.pi,np.pi,np.pi], repeat=True
	#stream=crappy.blocks.MeasureByStep(instronSensor,labels=['t(s)','signal','signal2'],freq=200)
	#stream_effort=crappy.blocks.StreamerComedi(sensor_effort,labels=['t(s)','angle','C(Nm)'],freq=2000)
	#stream=crappy.blocks.MeasureByStep(sensor,labels=['t(s)','T','T2','T3'],freq=100) #['t(s)','T']
	stream=crappy.blocks.Streamer(sensor,labels=['t(s)','T','T2','T3'])
	#stream=crappy.blocks.MeasureComediByStep(instronSensor, labels=['t(s)','V'], freq=1000.)
	#traction=crappy.blocks.SignalGenerator(path=[{"waveform":"sinus","time":100,"phase":0,"amplitude":1,"offset":0,"freq":2}],
												#send_freq=400,repeat=True)
	#torsion=crappy.blocks.SignalGenerator(path=[{"waveform":"triangle","time":50,"phase":0,"amplitude":5,"offset":-0.5,"freq":1}]
											#,send_freq=400,repeat=False,labels=['t(s)','signal'])
	
	#send_output=crappy.blocks.CommandComedi([cmd_traction,cmd_traction2])
	compacter=crappy.blocks.Compacter(4000)
	#compacter_effort=crappy.blocks.Compacter(2000)
	save=crappy.blocks.Saver("/home/corentin/Bureau/delete_2.txt")
	#graph=crappy.blocks.Grapher("dynamic",('t(s)','T'),('t(s)','T2'),('t(s)','T3')) #,('t(s)','T_mean')
	#graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','C(Nm)'))
	#graph_effort2=crappy.blocks.Grapher("dynamic",('t(s)','angle'))
	#save_effort=crappy.blocks.Saver("/home/corentin/Bureau/test_mesure_instron.txt")
	#graph2=crappy.blocks.Grapher("dynamic",('t(s)','ang(deg)'),('t(s)','dep(mm)'))
	#graph3=crappy.blocks.Grapher("dynamic",(0,4))
	
########################################### Creating links
	#crappy.links.Filter(labels=['dist(deg)'],mode="median",size=50)
	#condition=[crappy.links.Filter(labels=['V'],mode="median",size=50),crappy.links.Filter(labels=['t(s)'],mode="mean",size=50)]
	link1=crappy.links.Link()#crappy.links.Filter(labels=['T','T2','T3'], mode='mean', size=100),condition_calib(),condition_sub()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	
########################################### Linking objects
	stream.add_output(link1)
	#stream.add_output(link5)
	#stream_effort.add_output(link2)
	#traction.add_output(link2)
	#traction.add_output(link1)
	#torsion.add_output(link5)
	#compacter_effort.add_input(link2)
	#compacter_effort.add_output(link4)
	#compacter_effort.add_output(link6)
	#compacter_effort.add_output(link7)
	compacter.add_input(link1)
	#compacter2.add_input(link2)
	#send_output.add_input(link2)
	#compacter.add_output(link3)
	#compacter.add_output(link4)
	compacter.add_output(link5)
	#compacter.add_output(link6)
	#compacter.add_output(link7)
	
	save.add_input(link5)
	#save_effort.add_input(link6)
	#graph.add_input(link3)
	#graph_effort.add_input(link4)
	#graph_effort2.add_input(link7)
	#graph_stat.add_input(link5)
	#graph2.add_input(link4)
	#graph3.add_input(link7)
########################################### Starting objects
	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)
		
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution


########################################### Stopping objects

	#for instance in crappy.blocks.MasterBlock.instances:
		#instance.stop()

except KeyboardInterrupt:
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.stop()
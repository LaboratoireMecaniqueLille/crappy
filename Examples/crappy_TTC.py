import time
import numpy as np
import crappy
import pandas as pd
#crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances



class condition_signal(crappy.links.MetaCondition):
	def __init__(self,input_value_label):
		self.input_value_label=input_value_label
		
	def evaluate(self,value):
		value['signal']=value.pop(self.input_value_label)
		return value

class eval_strain(crappy.links.MetaCondition):
	def __init__(self,k):
		self.surface=110.74*10**(-6)
		self.I=np.pi*((25*10**-3)**4-(22*10**-3)**4)/32
		self.rmoy=((25+22)*10**(-3))/2
		self.size=20
		self.labels=['dist(deg)','def(%)','C(Nm)','F(N)']
		self.FIFO=[[] for label in self.labels]
		self.k=k #for testing
		
	def evaluate(self,value):
		#if self.k==1:
			#print value
		for i,label in enumerate(self.labels):
			#print self.FIFO[i]
			self.FIFO[i].insert(0,value[label])
			if len(self.FIFO[i])>self.size:
				self.FIFO[i].pop()
			result=np.mean(self.FIFO[i])
			value[label]=result
		value['tau(Pa)']=((value['C(Nm)']/self.I)*self.rmoy)
		value['sigma(Pa)']=(value['F(N)']/self.surface)
		value['eps_tot(%)']=np.sqrt((value['def(%)'])**2+((value['dist(deg)'])**2)/3.)
		#if self.k==1:
			#print value
		return value


if __name__ == '__main__':
	try:
	########################################### Creating objects
	# we measure the offset to have the correct value for def and dist
		instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2,3],gain=[0.01998,99660,0.0099856*2.,499.5]) # dist is multiplied by 2 to be correct
		offset=np.array([0.,0.,0.,0.])
		for i in range(100):
			for j in range(0,4,2):
				offset[j]+=instronSensor.getData(j)[1]/100.
		#offset-=np.array([0,1806,0,0.175])
		offset*=-1
	# end of the offset measure
		instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1,2,3],gain=[0.01998,99660,0.0099856*2.,499.5],offset=offset) # 10 times the gain on the machine if you go through an usb dux sigma
		#cmd_traction=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=1, range_num=0, gain=8*100, offset=0)
		#cmd_torsion=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=8*100/2., offset=0) # divide dist by 2 for the testing machine
		
		#with Labjack:
		cmd_traction=crappy.actuator.LabJackActuator(channel="TDAC2", gain=8*100, offset=0)
		cmd_torsion=crappy.actuator.LabJackActuator(channel="TDAC3", gain=8*100/2., offset=0)

	########################################### Initialising the outputs

		cmd_torsion.set_cmd(0)
		cmd_traction.set_cmd(0)
		time.sleep(0.5)
		cmd_torsion.set_cmd(0)
		cmd_traction.set_cmd(0)
		print "ready ?"
		raw_input()
	########################################### Creating blocks
		#comedi_output=crappy.blocks.CommandComedi([comedi_actuator])
		
		stream=crappy.blocks.MeasureByStep(instronSensor,labels=['t(s)','def(%)','F(N)','dist(deg)','C(Nm)'])
		#stream=crappy.blocks.StreamerComedi(instronSensor, labels=['t(s)','def(%)','F(N)','dist(deg)','C(Nm)'], freq=200.)

		#{"waveform":"detection","cycles":2}
		#{"waveform":"trefle","gain":0.0005,"cycles":1,"offset":[0.0002,-0.0002]},
												#{"waveform":"sablier","gain":0.0005,"cycles":1,"offset":[-0.0002,0.0002]},
												#{"waveform":"sablier","gain":0.0005,"cycles":1,"offset":[-0.0002,0.0002]},
												#{"waveform":"goto","position":[0,0]}
		multipath=crappy.blocks.MultiPath(path=[{"waveform":"circle","gain":0.0006,"cycles":200,"offset":[-0.0006,0]},
												#{"waveform":"goto","mode":"plastic_def","target":0.002,"position":[-10,0]}
												{"waveform":"detection","cycles":1}],
												send_freq=200,dmin=22,dmax=25,repeat=False)
		
		ttc=crappy.blocks.CommandComedi([cmd_traction,cmd_torsion],signal_label=['def(%)','dist(deg)'])
		#torsion=crappy.blocks.CommandComedi([cmd_torsion])
		
		
		
		compacter_data=crappy.blocks.Compacter(200)
		save=crappy.blocks.Saver("/home/corentin/Bureau/data_labjack_200_cycles.txt")
		graph_traction=crappy.blocks.Grapher("static",('sigma(Pa)','tau(Pa)'))
		graph_torsion=crappy.blocks.Grapher("static",('def(%)','dist(deg)'))
		#graph_torsion=crappy.blocks.Grapher("static",('t(s)','def(%)'))
		
		compacter_path=crappy.blocks.Compacter(200)
		save_path=crappy.blocks.Saver("/home/corentin/Bureau/data_out_labjack_200_cycles.txt")
		graph_path=crappy.blocks.Grapher("dynamic",('t(s)','def(%)'))
		graph_path2=crappy.blocks.Grapher("dynamic",('t(s)','dist(deg)'))
		#graph_torsion=crappy.blocks.Grapher("dynamic",('t(s)','C(Nm)'))
		#graph_stat=crappy.blocks.Grapher("dynamic",(0,2))
		#graph2=crappy.blocks.Grapher("dynamic",(0,3))
		#graph3=crappy.blocks.Grapher("dynamic",(0,4))
		
	########################################### Creating links
		
		link1=crappy.links.Link(eval_strain(k=1))
		link2=crappy.links.Link()
		link3=crappy.links.Link()
		link4=crappy.links.Link(eval_strain(k=2))
		link5=crappy.links.Link()
		link6=crappy.links.Link()
		link7=crappy.links.Link()
		link8=crappy.links.Link()
		link9=crappy.links.Link()
		link10=crappy.links.Link()
		link11=crappy.links.Link()
		
	########################################### Linking objects
		stream.add_output(link1)
		stream.add_output(link4)
		
		compacter_data.add_input(link1)
		compacter_data.add_output(link2)
		compacter_data.add_output(link3)
		compacter_data.add_output(link11)
		#compacter_data.add_output(link6)
		#compacter_data.add_output(link7)
		#compacter_data.add_output(link10)

		graph_traction.add_input(link2)
		graph_torsion.add_input(link3)
		save.add_input(link11)
		
		multipath.add_input(link4)
		multipath.add_output(link5)
		multipath.add_output(link8)
		#multipath.add_output(link9)
		
		
		compacter_path.add_input(link5)
		compacter_path.add_output(link6)
		compacter_path.add_output(link7)
		compacter_path.add_output(link10)
		
		save_path.add_input(link10)

		graph_path.add_input(link6)
		graph_path2.add_input(link7)
		
		ttc.add_input(link8)
		#torsion.add_input(link9)

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

	except (Exception,KeyboardInterrupt) as e:
		print "Exception in main :", e
		for instance in crappy.blocks._meta.MasterBlock.instances:
			try:
				instance.stop()
				print "instance stopped : ", instance
			except:
				pass
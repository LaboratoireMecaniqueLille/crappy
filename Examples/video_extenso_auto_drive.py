import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
import numpy as np
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances

class condition_transfo_chaine(crappy.links.MetaCondition):
	def __init__(self):
		data="None"
	def evaluate(self,value):
		if(value == 'error'):
			return {'center':0.}
		data=value['Py']
		data_chaine = filter(None,data.lstrip('[ ').rstrip(' ]').split(' '))
		for i in range(0,len(data_chaine)):
			data_chaine[i] = float(data_chaine[i])
		center = (float(data_chaine[0])+float(data_chaine[1]))/2. - 2048/2.
		return {'center':center}

class condition_comedi_instron(crappy.links.MetaCondition):
	def __init__(self):
		data="None"
		i=0
	def evaluate(self,value):
		data=value['Eyy(%)']
		data_chaine = (1.+(data/100.))*2/5
		return {'signal':data_chaine}


if __name__ == '__main__':
	try:
	########################################### Creating objects
		
		#instronSensor=crappy.sensor.ComediSensor(channels=[1,3],gain=[-3749.3,-3198.9*1.18],offset=[24,13])
		#biaxeTech1=crappy.technical.Biaxe(port='/dev/ttyS4')
		#biaxeTech2=crappy.technical.Biaxe(port='/dev/ttyS5')
		#biaxeTech3=crappy.technical.Biaxe(port='/dev/ttyS6')
		#biaxeTech4=crappy.technical.Biaxe(port='/dev/ttyS7')
		#axes=[biaxeTech1,biaxeTech2,biaxeTech3,biaxeTech4]
		comedi_actuator=crappy.actuator.ComediActuator(device='/dev/comedi0',subdevice=0,channel=0,range_num=0,gain=1,offset=0)
		comedi_actuator.set_cmd(0)
		time.sleep(0.5)
		comedi_actuator.set_cmd(0)

	########################################### Creating blocks
		
		#compacter_effort=crappy.blocks.Compacter(200)
		#save_effort=crappy.blocks.Saver("/home/biaxe/Bureau/Publi/effort.txt")
		#graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','F2(N)'),('t(s)','F4(N)'))
		
		compacter_extenso=crappy.blocks.Compacter(100)
		save_extenso=crappy.blocks.Saver("/home/erwan/Bureau/essai_relax.txt")
		graph_extenso=crappy.blocks.Grapher("dynamic",('t(s)','Eyy(%)','Px','Py'))
		
		#effort=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','F2(N)','F4(N)'],freq=200)

		extenso=crappy.blocks.VideoExtenso(camera="Ximea",numdevice=0,xoffset=0,yoffset=200,width=2048,height=512,white_spot=True,display=True,security=True)
		autoDrive=crappy.blocks.AutoDrive(crappy.actuator.CmDrive())
		comediActuator=crappy.blocks.CommandComedi([comedi_actuator])
  

	########################################### Creating links
		
		link1=crappy.links.Link()
		link2=crappy.links.Link()
		link6=crappy.links.Link()

		link3=crappy.links.Link()
		link4=crappy.links.Link()
		link5=crappy.links.Link()
  
		linkAuto1=crappy.links.Link(condition=[condition_transfo_chaine(),crappy.links.PID(P=3000., I=0, D=0, label_consigne='K', label_retour='center', consigne=0., outMin=-90000, outMax=90000,add_current_value=False)])
		linkExx1=crappy.links.Link(condition=condition_comedi_instron())
  
  
		
		
	########################################### Linking objects

		#effort.add_output(link1)
		#effort.add_output(link12)
		#effort.add_output(link6)
		
		#extenso.add_output(link2)
		#extenso.add_output(link22)
		extenso.add_output(link3)
		extenso.add_output(linkAuto1)
		extenso.add_output(linkExx1)
		#extenso2.add_output(link1)
		autoDrive.add_input(linkAuto1)
		comediActuator.add_input(linkExx1)
		#signalGenerator.add_input(link1)
		#signalGenerator.add_input(link2)
		#signalGenerator.add_output(link9)
		
		#signalGenerator_horizontal.add_input(link12)
		#signalGenerator_horizontal.add_input(link22)
		#signalGenerator_horizontal.add_output(link92)
		
		#biotens.add_input(link9)
		#biotens_horizontal.add_input(link92)

		#compacter_effort.add_input(link6)
		#compacter_effort.add_output(link7)
		#compacter_effort.add_output(link8)
		
		#save_effort.add_input(link7)
		
		#graph_effort.add_input(link8)
		
		compacter_extenso.add_input(link3)
		#compacter_extenso2.add_input(link1)
		compacter_extenso.add_output(link4)
		compacter_extenso.add_output(link5)
		#compacter_extenso2.add_output(link2)
		#compacter_extenso2.add_output(link6)
		
		save_extenso.add_input(link4)
		
		graph_extenso.add_input(link5)
		#graph_extenso2.add_input(link2)
		#save_extenso2.add_input(link6)
		
	########################################### Starting objects
		#print "top :",crappy.blocks._meta.MasterBlock.instances
		t0=time.time()
		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.set_t0(t0)

		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.start()
		
	########################################### Waiting for execution


	########################################### Stopping objects

	except (Exception,KeyboardInterrupt) as e:
		print "Exception in main :", e
		for instance in crappy.blocks._meta.MasterBlock.instances:
			try:
				instance.stop()
			except:
				pass
			

import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances


t0=time.time()

try:
########################################### Creating objects
	
	instronSensor=crappy.sensor.ComediSensor(channels=[1,3],gain=[-6725,-6000],offset=[-110,0])
	biaxeTech1=crappy.technical.Biaxe(port='/dev/ttyS4')
	biaxeTech2=crappy.technical.Biaxe(port='/dev/ttyS5')
	biaxeTech3=crappy.technical.Biaxe(port='/dev/ttyS6')
	biaxeTech4=crappy.technical.Biaxe(port='/dev/ttyS7')
	axes=[biaxeTech1,biaxeTech2,biaxeTech3,biaxeTech4]

########################################### Creating blocks
	
	compacter_effort=crappy.blocks.Compacter(200)
	save_effort=crappy.blocks.Saver("/home/biaxe/Bureau/effort.txt")
	graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','F(N)'),('t(s)','F3(N)'))
	
	compacter_extenso=crappy.blocks.Compacter(75)
	save_extenso=crappy.blocks.Saver("/home/biaxe/Bureau/extenso.txt")
	graph_extenso=crappy.blocks.Grapher("dynamic",('t(s)','Exx(%)'),('t(s)','Eyy(%)'))
	
	effort=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','F(N)','F3(N)'],freq=200)
	extenso=crappy.blocks.VideoExtenso(camera="Ximea",xoffset=400,yoffset=400,width=1000,height=1000,white_spot=True,labels=['t(s)','Exx(%)', 'Eyy(%)'],display=True)
	
	#pathGenerator=crappy.blocks.PathGenerator(send_freq=1000,waveform=["limit","limit","limit"],time_cycles=[0.6,1,3],phase=[0,0,0],lower_limit=[[0.05,'F(N)'],[0.05,'F(N)'],[0,None]],upper_limit=[[5.0,'Eyy(%)'],[4.0,'Eyy(%)'],[0,None]],repeat=True)
	#pathGenerator=crappy.blocks.SignalGenerator(send_freq=1000,waveform=["limit","limit","limit","limit"],time_cycles=[5,5,5,0.5],phase=[0,0,0,0],lower_limit=[[0.05,'F(N)'],[0.05,'F(N)'],[0.05,'F(N)'],[0.05,'F(N)']],upper_limit=[[5.0,'Eyy(%)'],[10.0,'Eyy(%)'],[20.0,'Eyy(%)'],[90,'F(N)']],repeat=False)
	
	
	signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"hold","time":3},
							{"waveform":"limit","cycles":3,"phase":0,"lower_limit":[50,'F(N)'],"upper_limit":[5,'Exx(%)']}],
							send_freq=400,repeat=True,labels=['t(s)','signal'])
	
	signalGenerator_horizontal=crappy.blocks.SignalGenerator(path=[{"waveform":"hold","time":3},
							{"waveform":"limit","cycles":3,"phase":0,"lower_limit":[0.5,'Eyy(%)'],"upper_limit":[5,'Eyy(%)']}],
							send_freq=400,repeat=True,labels=['t(s)','signal'])
	
	biotens=crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech1,biaxeTech2],speed=-500) # vertical
	biotens_horizontal=crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech3,biaxeTech4],speed=-500) #horizontal

########################################### Creating links
	
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link12=crappy.links.Link()
	link22=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	link9=crappy.links.Link()
	link92=crappy.links.Link()
	
########################################### Linking objects

	effort.add_output(link1)
	effort.add_output(link12)
	effort.add_output(link6)
	
	extenso.add_output(link2)
	extenso.add_output(link22)
	extenso.add_output(link3)

	signalGenerator.add_input(link1)
	signalGenerator.add_input(link2)
	signalGenerator.add_output(link9)
	
	signalGenerator_horizontal.add_input(link12)
	signalGenerator_horizontal.add_input(link22)
	signalGenerator_horizontal.add_output(link92)
	
	biotens.add_input(link9)
	biotens_horizontal.add_input(link92)

	compacter_effort.add_input(link6)
	compacter_effort.add_output(link7)
	compacter_effort.add_output(link8)
	
	save_effort.add_input(link7)
	
	graph_effort.add_input(link8)
	
	compacter_extenso.add_input(link3)
	compacter_extenso.add_output(link4)
	compacter_extenso.add_output(link5)
	
	save_extenso.add_input(link4)
	
	graph_extenso.add_input(link5)
	
########################################### Starting objects

	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)

	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.start()

########################################### Waiting for execution


########################################### Stopping objects

except :
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.join()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.stop()
	for axe in axes:
		axe.close()
		
import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances


t0=time.time()


#class condition_F(crappy.links.MetaCondition):
	#def __init__(self,test=False):
		#self.F_offset=0.1
		
	#def evaluate(self,value):
		#recv=self.external_trigger.recv(blocking=False) # first run is blocking, others are not

		#try:
			#self.new_coeff=recv['coeff'][0]
			##print "new_coeff :", self.new_coeff
		#except TypeError:
			#pass
		#if self.new_coeff!=self.coeff: # if coeff is changing
			#if self.new_coeff!=self.last_new_coeff: # if first change
				#self.t_init=time.time()
				#self.t1=self.t_init
				#self.last_new_coeff=self.new_coeff
			#self.t2=time.time()
			#if (self.t2-self.t_init)<self.delay:
				#self.coeff+=(self.new_coeff-self.last_coeff)*((self.t2-self.t1)/(self.delay))
			#else: # if delay is passed
				#self.coeff=self.new_coeff
				#self.last_coeff=self.coeff
			#self.t1=self.t2
		##print "coeff :", self.coeff
		#value['signal'][0]*=self.coeff
		#if self.test:
			#return None
		#else:
			#return value

#class condition_K(crappy.links.MetaCondition):
	#def __init__(self):
		#self.K=0
		#self.W = 18.*10**(-3) #largeur eprouvette
		#self.y = 3.*10**(-3) #distance de prise potentielle depuis centre eprouvette
		#self.a0_elec= 4.1*10**(-3) #longueur prefissure
		#self.e = 3.8*10**(-3) # epaisseur eprouvette
		#self.K1=8*10**6
		#self.F0=8000.
		#self.K0=self.F0/(2000.) # 2000 Newtons/Volt on the instron computer
		#self.FIFO=[]
		#self.size=120 # 120 cycles = 1 minute
		#if self.K0>4:
			#print "WARNING, K0 is too high for the USB-DUX D, please stop and modify your script"
		#self.first=True
		## self.V0=   ################################################################################################################################### Add here the v0 value if you restart the script
	#def evaluate(self,value):
		#self.FIFO.insert(0,value['t_agilent(s)'][0])
		#if len(self.FIFO)>self.size:
			#self.FIFO.pop()
		#median_value=np.median(self.FIFO)
		#if value['t_agilent(s)'][0] > 60000: ###################################################################################################################### delay before starting
			#if self.first:
				#self.first=False
				#self.V0= value['tension(V)'][0]
				#np.savetxt('/home/essais-2015-3/Bureau/V0.txt',[self.V0])
			#a= (2.*self.W/np.pi)*np.arccos(np.cosh(np.pi*self.y/(2.*self.W))/np.cosh(median_value/self.V0*np.arccosh(np.cosh(np.pi*self.y/(2.*self.W))/np.cos(np.pi*self.a0_elec*10**(-3)/(2.*self.W)))))
			#alpha = a/self.W #rapport longueur fissure sur largeur
			#Y = alpha**4*196.89980597-alpha**3*281.49618641+alpha**2*157.05615266-alpha*36.9122841+3.54991714
			#Fmax = self.K1/(Y*np.sqrt(3.1416*a))*self.e*self.W
			#if not(np.isnan(Fmax)):
				#self.K=(Fmax/self.F0)*self.K0
			#print "a, Fmax, K : ", a, Fmax, self.K
		#if self.K>self.K0:
			#print "WARNING, evaluation of K is wrong!"
			#self.K=self.K0
			
		#value['coeff'] = pd.Series((self.K), index=value.index)
		##print value
		#return value

try:
########################################### Creating objects
	
	instronSensor=crappy.sensor.ComediSensor(channels=[0],gain=[-48.8],offset=[0])
	t,F0=instronSensor.getData(0)
	print "offset=", F0
	instronSensor=crappy.sensor.ComediSensor(channels=[0],gain=[-48.8],offset=[-F0])
	biotensTech=crappy.technical.Biotens(port='/dev/ttyUSB0', size=30)

########################################### Creating blocks
	
	compacter_effort=crappy.blocks.Compacter(150)
	save_effort=crappy.blocks.Saver("/home/annie/Bureau/essais_paroi_video/temoin_effort_3.txt")
	graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','F(N)'))
	
	compacter_extenso=crappy.blocks.Compacter(90)
	save_extenso=crappy.blocks.Saver("/home/annie/Bureau/essais_paroi_video/temoin_extenso_3.txt")
	graph_extenso=crappy.blocks.Grapher("dynamic",('t(s)','Exx(%)'),('t(s)','Eyy(%)'))
	
	effort=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','F(N)'],freq=150)
	extenso=crappy.blocks.VideoExtenso(camera="Ximea",white_spot=False,labels=['t(s)','Exx(%)', 'Eyy(%)'],display=True)
	
	#signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"hold","time":0},
							#{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.05,'F(N)'],"upper_limit":[90,'Eyy(%)']}],
							#send_freq=400,repeat=False,labels=['t(s)','signal'])
	#example of path:[{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.05,'F(N)'],"upper_limit":[i,'Eyy(%)']} for i in range(10,90,10)]

	signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"limit","gain":1,"cycles":2,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[5,'Eyy(%)']},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[10,'Eyy(%)']},
							{"waveform":"hold","time":120},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[20,'Eyy(%)']},
							{"waveform":"hold","time":120},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[30,'Eyy(%)']},
							{"waveform":"hold","time":120},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[40,'Eyy(%)']},
							{"waveform":"hold","time":120},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[50,'Eyy(%)']},
							{"waveform":"hold","time":120},
							{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.02,'F(N)'],"upper_limit":[90,'F(N)']}],
	
							send_freq=5,repeat=False,labels=['t(s)','signal','cycle'])
	
	
	biotens=crappy.blocks.CommandBiotens(biotens_technicals=[biotensTech],speed=5)
	compacter_position=crappy.blocks.Compacter(5)
	save_position=crappy.blocks.Saver("/home/annie/Bureau/essais_paroi_video/temoin_position_3.txt")

########################################### Creating links
	
	link1=crappy.links.Link()
	link2=crappy.links.Link()
	link3=crappy.links.Link()
	link4=crappy.links.Link()
	link5=crappy.links.Link()
	link6=crappy.links.Link()
	link7=crappy.links.Link()
	link8=crappy.links.Link()
	link9=crappy.links.Link()
	link10=crappy.links.Link()
	link11=crappy.links.Link()
	
########################################### Linking objects

	effort.add_output(link1)
	effort.add_output(link6)
	
	extenso.add_output(link2)
	extenso.add_output(link3)

	signalGenerator.add_input(link1)
	signalGenerator.add_input(link2)
	signalGenerator.add_output(link9)
	
	biotens.add_input(link9)
	biotens.add_output(link10)

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
	
	compacter_position.add_input(link10)
	compacter_position.add_output(link11)
	
	save_position.add_input(link11)
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
		
#try:
	#while True:
		#print instronSensor.getData(0)[1]
		#time.sleep(0.1)
#except KeyboardInterrupt:
	#pass
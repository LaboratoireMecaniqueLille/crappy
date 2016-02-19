# coding: utf-8
from _meta import MasterBlock
import numpy as np
import time
import pandas as pd
from scipy import stats
from collections import OrderedDict
import copy
#import pickle
from ..links._link import TimeoutError
import multiprocessing
from sys import stdout

class MultiPath(MasterBlock):
	"""
	Children class of MasterBlock. Use it for traction-torsion testing.
	"""
	def __init__(self,path=None,send_freq=400,dmin=22,dmax=25,\
			  default_G=71*10**9,default_E=196*10**9,repeat=False):
		"""
This block is specific for use in traction-torsion testing. You need 
to define a path to follow, with the available waveform. Unlike the 
SignalGenerator block, as we don't need time synchronisation in this 
case, the link beween one step and the next will be done smoothly and 
automatically, even if there is a gap.
default_G and default_E will only be used for plasticity evaluation in
case of "goto" waveform, and if the evaluated E and G are not good 
enough in plasticity detection. This can happend if your are close to
the axis, e.g is the def or dist stay close to 0. To avoid this 
phenomenom, if the evaluated vectors are too close of the axis, they 
will be rotated.

Parameters
----------
path : list of dict
	Each dict must contain parameters for one step.
	
	See Examples section below.
	
		waveform : {‘detection’,’goto’,’trefle’,’sablier’,’circle’,\ 
			‘traction’,’torsion’,’proportionnal’}
			Shape of your signal, for every step. Possible values are :
			
			**detection** is the plasticity surface detection.
			
			**goto** get you to a certain point in the def-dist referentiel :
			You can specify *mode* : *plastic_def* to apply a plastic load.
			In this case, you will move in direction of **position** until 
			**target** def is reached.
		time : int or float or None.
			Time before change of step, for every step. If None, means infinite.
		cycles : int or float or None (default).
			Number of cycles before change of step, for every step. If None, means infinite.
		gain : int or float.
			Amplitude of your signal. WARNING : a gain of 1 will result in 100% deformation.
		offset: tuple of int of tuple of float
			Offset of your signal.
			
send_freq : int or float , default = 400
	Loop frequency. Use this parameter to avoid over-use of processor.
	
dmin : int or float, default = 22
	value of the internal diameter of the test specimen, in mm.
	
dmax : int or float, default = 25
	value of the external diameter of the test specimen, in mm.
	
default_G : int or float, default = 71*10**9
	value of the default shear modulus, in Pa.
	
default_E : int or float, default = 196*10**9
	value of the default Young modulus, in Pa.
	
repeat : Boolean, default=False
	Set True is you want to repeat your sequence forever.

Returns
-------
dict : OrderedDict


	def(%) : 
		output signal for traction
	
	dist(deg) : 
		output signal for torsion
	
	def_plast(%) : 
		evaluated plastic def, if evaluated
	
	E(Pa) : 
		Young modulus
	
	G(Pa) : 
		shear modulus
	
	status : 
		Status of the plasticity detection, formated x.y
	
		- x : number of the current branch 
		- y : substep 
		
			* 0 : just starting, eliminating the first points
			
			* 1 : evaluating E and G
			
			* 2 : detecting plasticity
			
			* 3 : platicity detected
			
			* -1 : plasticity surface detected
	
	
	relative_eps_tot : 
		total deformation, relative to the starting point. Used for 
		plasticity detection.


Examples
--------
>>> SignalGenerator(path=[{"waveform":"detection","cycles":1},
{"waveform":"goto","mode":"total_def","position":[0,0]},
{"waveform":"goto","mode":"plastic_def","target":0.002,"position":[-10,0]},
{"waveform":"trefle","gain":0.001,"cycles":1,"offset":[0.001:-0.002]},
{"waveform":"traction","gain":0.001,"cycles":0.25,"offset":[-0.001:0.002]}],
send_freq=400,dmin=22,dmax=25,default_G=71*10**9,default_E=196*10**9,repeat=False)

In this example we displayed some possibilities of waveform.
Every dict contains informations for one step.
The requiered informations depend on the type of waveform you need.
		""" 
		print "MultiPath!"
		self.path=path # list of list or arrays
		self.nb_step=len(path)
		self.send_freq=send_freq
		self.repeat=repeat
		self.step=0
		self.surface=(np.pi*((dmax/2.)**2-(dmin/2.)**2))*10**-6    #110.74*10**(-6)
		self.offset=5*10**-6#5*10**(-6)
		self.R_eps=0.0001 # in (0.00005)
		self.D_eps=0.0005 # in  (0.00055)
		self.normal_speed=6.6*10**(-4) #in /s
		self.detection_speed=6.6*10**(-5) # in /s
		self.speed=self.normal_speed
		self.plastic_offset=2*10**(-5)
		self.last_t_goto=0
		self.rmoy=(dmax+dmin)*10**(-3)/2
		self.I=np.pi*((dmax*10**-3)**4-(dmin*10**-3)**4)/32
		self.plastic_def=0
		self.status=-1
		#self.E=190*10**9
		self.default_G=default_G #71*10**9
		self._relative_total_def=0
		self.default_E=default_E #196844*10**6 #165546471680.81555 for duplex
		self.G=self.default_G #69663620164.045624
		self.E=self.default_E
		
	def send(self,traction,torsion):
		Array=OrderedDict(zip(['t(s)','def(%)','dist(deg)','def_plast(%)','E(Pa)','G(Pa)','status','relative_eps_tot'],[time.time()-self.t0,traction,torsion,self.plastic_def,self.E,self.G,self.status,self._relative_total_def]))
		#print "array : ",Array
		try:
			for output in self.outputs:
				output.send(Array)
		except TimeoutError:
			raise
		except AttributeError: #if no outputs
			pass
	
	def return_elastic(self):
		print "return to elastic area"
		self.initial_total_def=self.total_def
		self.initial_position=self.position
		if self.total_def>0.0011:
			while self.total_def>(self.initial_total_def-0.001): # go back from 0.1%
				self.goto([0,0],mode="towards")
				self.get_position()
			
	def detection(self):
		self.detection_step=0
		self.first_step_of_detection=True
		#self.speed=self.detection_speed
		self.plastic_def=0
		self.denoiser=0
		self.FIFO=[]
		self.first_of_branch=True
		while self.detection_step<16: # while we doesn't have 16 points for the plasticity surface
			self.speed=self.detection_speed
			self.get_position()
			#self.initial_position=self.position 
			#print self.position, self.initial_position
			if self.first_step_of_detection:
				self.status=0
				tilt=False
				print "detecting central position and evaluating vectors..."
				self.central_position=self.position
				self.central_effort=self.effort
				try:
					first_vector=np.subtract(self.initial_position,self.position)/np.linalg.norm(np.subtract(self.initial_position,self.position))
					ratio=abs(max(first_vector[0],first_vector[1])/min(first_vector[0],first_vector[1]))
					if ratio>5 or ratio<0.2: # if first vector is too close to axis (approx 10-15deg), tilt it.
						tilt=True
				except AttributeError: # if initial_position is not defined, means this is the first detection
					first_vector=[1,0]
					tilt=True
				a0=np.angle(np.complex(first_vector[0],first_vector[1]))
				#angles=np.arange(a0,a0+2*np.pi,np.pi/16.) # create 16 vectors equali oriented.
				angles=np.arange(a0,a0+2*np.pi,np.pi/8.) # create 16 vectors equali oriented.
				angles[1::2]+=np.pi #reorganized vectors
				if tilt:
					angles+=np.pi/16.
				self.target_positions=[np.cos(angles)*10,np.sin(angles)*10*np.sqrt(3)]# normalized positions, multiplied by 10 to be unreachable
				self.first_step_of_detection=False
				self.detection_substep=0
			#print first_vector
			#self._relative_total_def=np.sqrt((self.position[0]-self.central_position[0])**2+((self.position[1]-self.central_position[1])**2)/3.)
			if self.detection_substep==0: # if going toward plasticity
				try:
					self.goto([self.target_positions[0][self.detection_step],self.target_positions[1][self.detection_step]],mode="towards") # move a little to detetc the plasticity surface
					#print self.target_positions
					self.get_position()
					self._relative_total_def=np.sqrt((self.position[0]-self.central_position[0])**2+((self.position[1]-self.central_position[1])**2)/3.)
					if self.status>self.detection_step:
						self.FIFO.insert(0,np.sqrt((self.position[0]-self.eps0-(self.effort[0]-self.F0)/self.E)**2+((self.position[1]-self.gam0-(self.effort[1]-self.C0)/self.G)**2)/3.))
						if len(self.FIFO)>10:
							self.FIFO.pop()
						self.plastic_def=np.mean(self.FIFO)
					if self._relative_total_def<self.R_eps:
						if self.status != self.detection_step+0.0:
							print "eliminating first points"
							self.status=self.detection_step+0.0
					elif self._relative_total_def<self.R_eps+self.D_eps: # eval E and G
						if self.first_of_branch:
							self.eps0=self.position[0]
							self.F0=self.effort[0]
							self.gam0=self.position[1]
							self.C0=self.effort[1]
							self.eps=[]
							self.F=[]
							self.gam=[]
							self.C=[]
							self.first_of_branch=False
						self.eps.append(self.position[0]-self.eps0)
						self.F.append(self.effort[0]-self.F0)
						self.gam.append(self.position[1]-self.gam0)
						self.C.append(self.effort[1]-self.C0)
						if len(self.eps)>15:
							self.E, intercept, self.r_value_E, p_value, std_err = stats.linregress(self.eps,self.F)
							self.G, intercept, self.r_value_G, p_value, std_err = stats.linregress(self.gam,self.C)
							if self.r_value_E<0.99:
								self.E=self.default_E #165*10**9
							if self.r_value_G<0.99:
								self.G=self.default_G #69*10**9
						if self.status != self.detection_step+0.1:
							self.status=self.detection_step+0.1
							print "evaluating E and G"
					elif self.plastic_def<self.plastic_offset:
						#self.plastic_def=np.sqrt((self.position[0]-self.eps0-(self.effort[0]-self.F0)/self.E)**2+((self.position[1]-self.gam0-(self.effort[1]-self.C0)/self.G)**2)/3.)
						#self.eps=[]
						#self.F=[]
						#self.gam=[]
						#self.C=[]
						if self.status != self.detection_step+0.2:
							self.status=self.detection_step+0.2
							print "detecting plasticity..."
					else: # if plasticity
						self.denoiser+=1
						#self.plastic_def=np.sqrt((self.position[0]-self.eps0-(self.effort[0]-self.F0)/self.E)**2+((self.position[1]-self.gam0-(self.effort[1]-self.C0)/self.G)**2)/3.)
						if self.denoiser>5: # ensure 5 measured points in plasticity to avoid noise
							self.detection_substep=1
							self.first_of_branch=True
							self.status=self.detection_step+0.3
							self.denoiser=0
							self.FIFO=[]
							print "Plasticity detected !"
							#print type([self.E,self.G,self.eps,self.F,self.gam,self.C])
							#outfile=open("/home/corentin/Bureau/data_"+str(self.detection_step), "wb" )
							#pickle.dump([self.E,self.G,self.eps,self.F,self.gam,self.C], outfile)
							#np.save('/home/corentin/Bureau/data_'+str(self.detection_step),np.asarray([self.E,self.G,self.eps,self.F,self.gam,self.C]))
							print "E, G : ", self.E, self.G
							print "coeffs E, G : ", self.r_value_E, self.r_value_G
							#print "eps: ", self.eps
							#print "F :",self.F
							#print "dist :", self.gam
							#print "C :", self.C
				except ZeroDivisionError:
					print "E and/or G are not defined, please check your parameters"
			else: # if detection_substep==1, going back to center
				#self.E=0
				#self.G=0
				print "going back to center"
				self.speed=self.normal_speed/5.
				self.status=self.detection_step+0.3
				self.goto(self.central_position,mode="absolute")
				self.detection_substep=0
				self.detection_step+=1
				self.plastic_def=0
				print "moving to next vector : ", self.detection_step
		self.speed=self.normal_speed # setting back the normal speed
		self.status=-1
		print "plasticity surface detected"

	def get_position(self): # get position and eval total stress
		self.Data=self.inputs[0].recv()
		self.position=[self.Data['def(%)'],self.Data['dist(deg)']]
		self.effort=[self.Data['sigma(Pa)'],self.Data['tau(Pa)']]
		self.total_def=self.Data['eps_tot(%)']

	def goto(self,target,mode="towards"): # go to absolute position, use as a substep in main loop
		if mode=="towards":
			#########################print "going to : " ,target
			if time.time()-self.last_t_goto>1:
				self.last_t_goto=time.time()
			if np.linalg.norm(np.subtract(target,self.position))>self.offset:
				self.vector=np.subtract(target,self.position)/np.linalg.norm(np.subtract(target,self.position))
				#self.traction=self.position[0]+self.speed*self.vector[0]
				#self.torsion=self.position[1]+self.speed*self.vector[1]
				t=time.time()
				if np.linalg.norm(np.subtract(target,self.position))<20*self.offset:
					#coeff=0.5*np.subtract(target,self.position)
					coeff=0.2*self.speed*self.vector*(t-self.last_t_goto)
					#print "a"
				else:
					coeff=self.speed*self.vector*(t-self.last_t_goto)
				#print "vector : " , self.vector
				#print "delta t : ", (t-self.last_t_goto)
				#print "speed : ", self.speed
				#try :
					##print "1111111111111111111111111111111111111111111"
					#self.traction+=self.speed*self.vector[0]*(t-self.last_t_goto)
					#self.torsion+=self.speed*self.vector[1]*(t-self.last_t_goto)
				#except AttributeError:
					##print "222222222222222222222222222222222222222222222222222"
					#self.traction=self.position[0]+self.speed*self.vector[0]*(t-self.last_t_goto)
					#self.torsion=self.position[1]+self.speed*self.vector[1]*(t-self.last_t_goto)
				try:
					self.traction+=coeff[0]
					self.torsion+=coeff[1]
				except AttributeError:
					self.traction=self.position[0]+coeff[0]
					self.torsion=self.position[1]+coeff[1]
				#if np.linalg.norm(np.subtract(target,self.position))<np.linalg.norm([self.traction,self.torsion]):
					#[self.traction,self.torsion]=self.position+1.5*np.subtract(target,self.position)
				self.last_t_goto=t
				#print "sending towards : ", self.traction, self.torsion
				self.send(self.traction,self.torsion)
		elif mode=="absolute":
			t0=time.time()
			#self.vector=np.subtract(target,self.position)/np.linalg.norm(np.subtract(target,self.position))
			while np.linalg.norm(np.subtract(target,self.position))>self.offset:
				#print "moving"
				self.vector=np.subtract(target,self.position)/np.linalg.norm(np.subtract(target,self.position))
				t=time.time()
				#if np.linalg.norm(np.subtract(target,self.position))<np.linalg.norm(self.speed*self.vector*(t-t0)):
				if np.linalg.norm(np.subtract(target,self.position))<20*self.offset:
					#coeff=0.5*np.subtract(target,self.position)
					coeff=0.2*self.speed*self.vector*(t-t0)
					#print "a"
				else:
					coeff=self.speed*self.vector*(t-t0)
				#print self.vector, self.position, target
				#self.traction=self.position[0]+self.speed*self.vector[0]
				#self.torsion=self.position[1]+self.speed*self.vector[1]
				####coeff=[val if np.linalg.norm(val)>self.offset/3. else 0 for val in coeff]
				try:
					self.traction+=coeff[0]
					self.torsion+=coeff[1]
				except AttributeError:
					self.traction=self.position[0]+coeff[0]
					self.torsion=self.position[1]+coeff[1]
				#print "originale position : ",coeff, np.linalg.norm(np.subtract(target,self.position)), np.linalg.norm(coeff) #,self.position ,self.traction,self.torsion,coeff
				#print np.linalg.norm(np.subtract(target,self.position)), np.linalg.norm([self.traction,self.torsion])
				#if np.linalg.norm(np.subtract(target,self.position))<np.linalg.norm([self.traction,self.torsion]): # if evaluated next point is too far from target
					##print "correcting "
					#diff=np.subtract(target,self.position)
					#self.traction=self.traction+diff[0]-self.speed*self.vector[0]*(t-t0)
	  				#self.torsion=self.torsion+diff[1]-self.speed*self.vector[1]*(t-t0)
					#print "corrected position : ", self.traction,self.torsion, self.position, np.subtract(target,self.position)
				t0=t
				#print "sending : ",self.traction,self.torsion
				#self.traction*=100.
				#self.last_traction,self.last_torsion=self.traction,self.torsion
				self.send(self.traction,self.torsion)
				self.get_position()
	
	def function_trefle(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(9.7*self.gain)))
		if i==0:
			self.theta=0
		def_=self.gain*np.sin(2*self.theta)*np.sin(self.theta+0.75*np.pi)+self.start_offset[0]
		dist=np.sqrt(3)*self.gain*np.sin(2*self.theta)*np.sin(self.theta+np.pi/4)+self.start_offset[1]
		return [def_,dist]
	
	def function_sablier(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(7.55*self.gain)))
		if i==0:
			self.theta=0
		def_=0.8*self.gain*np.sin(2*self.theta)+self.start_offset[0]
		dist=0.8*np.sqrt(3)*self.gain*np.sin(self.theta)+self.start_offset[1]
		return [def_,dist]
	
	def function_circle(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(2*np.pi*self.gain)))
		if i==0:
			self.theta=0
		def_=self.gain*np.cos(self.theta)+self.start_offset[0]
		dist=np.sqrt(3)*self.gain*np.sin(self.theta)+self.start_offset[1]
		return [def_,dist]
	
	def function_traction(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(2*np.pi*self.gain)))
		if i==0:
			self.theta=0
		def_=self.gain*np.sin(self.theta)+self.start_offset[0]
		dist=0+self.start_offset[1]
		return [def_,dist]
	
	def function_torsion(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(2*np.pi*self.gain)))
		if i==0:
			self.theta=0
		def_=0+self.start_offset[0]
		dist=self.gain*np.sin(self.theta)+self.start_offset[1]
		return [def_,dist]
	
	def function_proportionnal(self,i=1):
		self.theta=abs(((time.time()-self.t_init)*(2*np.pi*self.normal_speed)/(2*np.pi*self.gain)))
		if i==0:
			self.theta=0
		def_=self.gain*np.sin(self.theta)+self.start_offset[0]
		dist=self.gain*np.sin(self.theta)+self.start_offset[1]
		return [def_,dist]
	
	#def function_square(self): # not implemented
		#self.theta=((time.time()-self.t_init)*(self.normal_speed))
		#def_=self.gain*np.cos(self.theta)
		#dist=np.sqrt(3)*self.gain*np.sin(self.theta)
		#return [def_,dist]
	
	
	def main(self): ######### WIP
		try:
			self.t_init=self.t0
			#self.last_t=self.t0
			#self.first_of_branch=True
			#for i in range(10):
				#self.send(0,0) # for testing only
			#self.path=path
			#self.nb_step=len(path)
			self.step=0
			self.get_position()
			while self.step<self.nb_step:
				current_step=self.path[self.step] 
				print "step number : ", self.step
				if current_step["waveform"]=="detection":
					self.return_elastic()
					self.cycles=current_step["cycles"]
					for cycle in range(self.cycles):
						print "cycle number : ", cycle
						self.detection()
				elif current_step["waveform"]=="goto":
					if current_step["mode"]=="total_def":
						self.goto(current_step["position"],mode="absolute")
					elif current_step["mode"]=="plastic_def":
						FIFO=[]
						self.get_position()
						eps0=self.position[0]
						F0=self.effort[0]
						gam0=self.position[1]
						C0=self.effort[1]
						if current_step["position"][0]==0: #torsion pure
							value=abs(self.position[1]-gam0-(self.effort[1]-C0)/self.default_G)
						elif current_step["position"][1]==0: #traction pure
							value=abs(self.position[0]-eps0-(self.effort[0]-F0)/self.default_E)
						else : # composition
							value=np.sqrt((self.position[0]-eps0-(self.effort[0]-F0)/self.default_E)**2+((self.position[1]-gam0-(self.effort[1]-C0)/self.default_G)**2)/3.)
						FIFO.insert(0,value)
						while np.mean(FIFO)<current_step["target"]:
							self.speed=self.normal_speed/10.
							self.goto(current_step["position"],mode="towards")
							self.get_position()
							if current_step["position"][0]==0: #torsion pure
								value=abs(self.position[1]-gam0-(self.effort[1]-C0)/self.default_G)
							elif current_step["position"][1]==0: #traction pure
								value=abs(self.position[0]-eps0-(self.effort[0]-F0)/self.default_E)
							else : # composition
								value=np.sqrt((self.position[0]-eps0-(self.effort[0]-F0)/self.default_E)**2+((self.position[1]-gam0-(self.effort[1]-C0)/self.default_G)**2)/3.)
							FIFO.insert(0,value)
							if len(FIFO)>10:
								FIFO.pop()
							stdout.write("\rplastic def : %5.5f %% " % (np.mean(FIFO)*100))
							stdout.flush()
						print "\n"
						self.speed=self.normal_speed
				else:
					try:
						self.waveform=current_step["waveform"]
						self.gain=current_step["gain"]
						self.cycles=current_step["cycles"]
						self.start_offset=current_step["offset"]
					except KeyError as e:
						print "You didn't define parameter %s for step number %s" %(e,self.step)
						raise
					if self.waveform=='trefle':
						self.f=self.function_trefle
					elif self.waveform=='sablier':
						self.f=self.function_sablier
					elif self.waveform=='circle':
						self.f=self.function_circle
					elif self.waveform=='traction':
						self.f=self.function_traction
					elif self.waveform=='torsion':
						self.f=self.function_torsion
					elif self.waveform=='proportionnal':
						self.f=self.function_proportionnal
					#elif self.waveform=="surface_detection":
						#self.
					#elif self.waveform=='square':
						#self.f=self.function_square
					else:
						raise Exception('not an acceptable waveform for step number %s' %self.step)
					self.theta=0
					self.get_position()
					initial_step_position=self.position
					if np.linalg.norm(np.subtract([self.f(0)],self.position))>self.offset: # if not in starting position
						print "getting into starting position..."
						#self.goto([x + y for x, y in zip(self.f(0), initial_step_position)],mode="absolute")
						self.goto(self.f(0),mode="absolute")
						self.t_init=time.time()
					print "starting..."
					while self.theta<=2*np.pi*self.cycles:
						stdout.write("\r%d %% done" % (100*(self.theta)/(2*np.pi*self.cycles)))
						stdout.flush()
						self.get_position()
						#target=[x + y for x, y in zip(self.f(), initial_step_position)]
						target=self.f()
						self.send(target[0],target[1])
					print "\n"
				self.step+=1
				if self.repeat and self.step==self.nb_step:
					self.step=0
					
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in Multipath : " ,e
	#totale_distance :
		#trefle : 9.7A
		#sablier : 7.55A
		#cercle : 2PI*A
	#x=[]
	#y=[]
	#t0=time.time()
	#theta=0
	#while theta < 2*np.pi:
		#theta=((time.time()-t0)*(2*np.pi)/(2*np.pi)) # =Delta_t*2Pi*v/total_distance
		#print theta
		#a,b=self.f()
		#x.append(a)
		#y.append(b)
	#print "finished in : ", (time.time()-t0)
	#plt.plot(x,y)
	#plt.show()
		
#######   Write Crappy:
	#def main(self): 
		#self.last_t=self.t0
		#self.first_of_branch=True
		#xc=np.asarray([1,0,0,1])
		#xr=np.asarray([0,0,1,1,0,1])+max(xc)+0.2
		#xa=np.asarray([0,0,1,1,0,1,1])+max(xr)+0.2
		#xp1=np.asarray([0,0,1,1,0,1,1])+max(xa)+0.2
		#xp2=np.asarray([0,0,0,1,1,0,1])+max(xp1)+0.2
		#xy=np.asarray([0,0.5,0,1])+max(xp2)+0.2
		#x=np.concatenate((xc,xr,xa,xp1,xp2,xy))
		#x/=max(x)
		#x-=0.5

		#yc=np.asarray([1,1,0,0])
		#yr=np.asarray([0,1,1,0.5,0.5,0])
		#ya=np.asarray([0,1,1,0.5,0.5,0.5,0])
		#yp1=np.asarray([0,1,1,0.5,0.5,0.5,1])
		#yp2=np.asarray([1,0,0.5,0.5,1,1,1])
		#yy=np.asarray([1,0.5,0,1])
		#y=np.concatenate((yc,yr,ya,yp1,yp2,yy))
		#self.get_position()
		#x1=np.empty(0)
		#y1=np.empty(0)
		#for k in range(len(x)-1):
			#x1=np.concatenate((x1,np.linspace(x[k],x[k+1],20)))
			#y1=np.concatenate((y1,np.linspace(y[k],y[k+1],20)))
		##x1=np.linspace(min(x),max(x),1000)
		##y1=np.interp(x1,x,y)
		##plt.plot(x,y,'+b');plt.plot(x1,y1,'r');plt.xlim(-1,1);plt.ylim(-0.1,1.1);plt.show()
		#for step in range(len(x1)):
			#for l in range(10):
				#self.get_position()
				#time.sleep(0.001)
			##self.get_position()
			#self.send(y1[step]/100.,x1[step])
		#while True:
			#time.sleep(0.01)
			#self.get_position()
			#self.send(y1[-1]/100.,x1[-1])
			
######## for batman
	#def main(self):  # for batman
		#import numpy as np
		#import matplotlib.pyplot as plt
		#from skimage import measure
		#import skimage.io
		#self.last_t=self.t0
		#self.first_of_branch=True
		#img=skimage.io.imread("/home/corentin/Bureau/projets/crappy_TTC/batman_logo_by_satanssidekick-d60qtoz.png")
		#img=img>128
		#contours = measure.find_contours(img[::,::,0], 0.8)
		#contours=np.transpose(contours[0])
		#y1=(contours[0])/max(contours[0])
		#x1=((contours[1])/max(contours[1]))/3
		#x1-=np.mean(x1)
		#y1-=np.mean(y1)
		#time.sleep(1)
		#tor=0
		#tra=0
		#for i in range(50):
			#self.get_position()
		#for i in range(1000):
			#tor+=y1[0]/1000.
			#tra+=x1[1]/1000.
			#self.send(tor/100.,tra)
			#self.get_position()
			#time.sleep(0.02)
		#for k in range(3):
			#for step in range(len(x1)):
				#self.get_position()
				#time.sleep(0.02)
				#self.get_position()
				#self.send(y1[step]/100.,x1[step])
		#while True:
			#time.sleep(0.01)
			#self.get_position()
			#self.send(y1[-1]/100.,x1[-1])
		
		
		
########### trefle:
#theta=np.arange(0,2*np.pi,1*10**-4)
#def_=A*np.sin(2*theta)*np.sin(theta+0.75*np.pi)
#dist=np.sqrt(3)*A*np.sin(2*theta)*np.sin(theta+np.pi/4)
#sub_path=zip(def_,dist)
#for i in range(len(sub_path)):
	#self.get_position()
	#self.goto([sub_path[0],sub_path[1]],mode="absolute")
	#import time
	#import numpy as np
	#import matplotlib.pyplot as plt
	

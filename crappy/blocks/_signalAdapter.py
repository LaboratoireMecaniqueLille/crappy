from _meta import MasterBlock
import pandas as pd
import time
import itertools

class SignalAdapter(MasterBlock):
	"""Multiply the signal by a coefficient"""
	def __init__(self,initial_coeff=0,delay=5,send_freq=800,labels=['t(s)','signal']):
		"""
SignalAdapter(initial_coeff=0,delay=5,labels=['t(s)','signal'])

Multiply input signal by an input coefficient.

Parameters
----------
acquisition_step : int
	Number of values to save in each data-stream before returning the array.
initial_coeff : float, default = 0
	Initial multiplying coefficient.
delay : float, default = 5
	Duration of the transistions between 2 coefficients.
labels : list of string, default = ['t(s)','signal']
	Labels of output signal, in a pandas.DataFrame() format.
	
Returns:
--------
Panda Dataframe.

		"""
		print "signal adapter!"
		self.coeff=initial_coeff
		self.last_coeff=initial_coeff
		self.delay=delay
		self.labels=labels
		self.send_freq=send_freq
		
	def main(self):
		try:
			#last_signal=0
			#new_coeff=0
			first=True
			#Data=pd.DataFrame()
			last_t=self.t0
			while True:
				while time.time()-last_t<1./self.send_freq:
					time.sleep(1./(100*self.send_freq))
				last_t=time.time()	
				#a=time.time()
				#print "total time: ", (a-last_a)
				#last_a=a
				if first:
					first=False
					for input_ in self.inputs:
					#print "1"
						recv=input_.recv(blocking=True)
					try:
						new_coeff=recv['coeff'].values[0]
						#print "new coeff!"
					except KeyError:
						last_signal=recv['signal'].values[0]
						t=recv['t(s)'].values[0]
						#print "last signal !"
					except:
						pass
				for input_ in self.inputs:
					#print "1"
					recv=input_.recv(blocking=False)
					try:
						new_coeff=recv['coeff'].values[0]
						#print "new coeff!"
					except KeyError:
						last_signal=recv['signal'].values[0]
						t=recv['t(s)'].values[0]
						#print "last signal !"
					except:
						pass
				#a3=time.time()
				#aaa=list(itertools.chain.from_iterable([Data,rec]))
				#print aaa
				#Data=pd.concat(rec,ignore_index=True)
				#a4=time.time()
				#a2=time.time()
				#print "recv time :", (a3-a2)
				#print "concat time :", (a4-a3)
				#last_signal_index = (Data['signal']).last_valid_index()
				#new_coeff_index=(Data['coeff']).last_valid_index()
				#first_coeff_index=(Data['coeff']).first_valid_index()
				#first_signal_index=(Data['signal']).first_valid_index()
				
				#last_signal = Data['signal'][last_signal_index]
				#new_coeff=Data['coeff'][new_coeff_index]
				#first_coeff=Data['coeff'][first_coeff_index]
				#first_signal=Data['signal'][first_signal_index]
				#if new_coeff_index!=new_coeff_index and last_signal_index!=first_signal_index: # clean old data
					#Data=Data[min(new_coeff_index,last_signal_index):]
				#a3=time.time()
				#print "update DB time :", (a3-a2)
				if new_coeff!=self.coeff: # if coeff is changing
					if self.coeff==self.last_coeff: # if first change
						t_init=time.time()
						t1=t_init
					t2=time.time()
					if (t2-t_init)<self.delay:
						self.coeff+=(new_coeff-self.last_coeff)*((t2-t1)/(self.delay))
					else: # if less than 1% difference
						self.coeff=new_coeff
					t1=t2
				if new_coeff==self.coeff: # if coeff=new_coeff, update the last_coeff
					self.last_coeff=self.coeff
				#a4=time.time()
				#print "eval coeff time :", (a4-a3)
				Array=pd.DataFrame([[t,last_signal*self.coeff]],columns=self.labels)
				try:
					for j in range(len(self.outputs)):
						self.outputs[j].send(Array)
				except:
					pass
				#a5=time.time()
				#print "sending time :", (a5-a4)
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in SignalAdapter : ", e
			raise
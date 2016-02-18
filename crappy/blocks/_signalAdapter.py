# coding: utf-8
from _meta import MasterBlock
import pandas as pd
import time
import itertools
import os

class SignalAdapter(MasterBlock):
	"""DEPRECATED. Multiply the signal by a coefficient"""
	def __init__(self,initial_coeff=0,delay=5,send_freq=800,labels=['t(s)','signal']):
		"""
SignalAdapter(initial_coeff=0,delay=5,labels=['t(s)','signal'])

DEPRECATED: use a link condition instead. See Examples/fissuration_v2.py

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
		print "WARNING : the signal adapater block is deprecated"
		print "signal adapter!"
		self.coeff=initial_coeff
		self.last_coeff=initial_coeff
		self.delay=delay
		self.labels=labels
		self.send_freq=send_freq
		
	def main(self):
		try:
			print "signal adapter!", os.getpid()
			#last_signal=0
			#new_coeff=0
			first=True
			#Data=pd.DataFrame()
			last_t=self.t0
			#i=1
			#delta=0
			t_sleep=0
			t_recv=0
			t_calc=0
			t_send=0
			t_total=0
			k=1
			count=0
			block=[True,True]
			while True:
				t_1=time.time()
				while time.time()-last_t<1./self.send_freq:
					#time.sleep(1./(100*self.send_freq))
					pass
				last_t=time.time()	
				t_sleep=max(t_sleep,last_t-t_1)
				#a=time.time()
				#print "total time: ", (a-last_a)
				#last_a=a
				#t_test=0
				#t_coeff=0
				
				if first:
					for num,input_ in enumerate(self.inputs):
						#print "1"
						#print input_
						recv=input_.recv()
						#print "2"
						try:
							#print "new coeff ?"
							new_coeff=recv['coeff'].values[0]
							#print block
							block[num]=False
							#print block
							#print "new coeff!"
						except KeyError:
							last_signal=recv['signal'].values[0]
							t=recv['t(s)'].values[0]
							block[num]=True
							print "last signal !"
						except Exception as e:
							print e
							pass
				else: # if not the first time
					#last_signal=None
					#while last_signal is None:
						#count+=1
					for num,input_ in enumerate(self.inputs):
						#print block
						#print "11", block[num], input_
						recv=input_.recv(blocking=block[num])
						#print "21"
						try:
							new_coeff=recv['coeff'].values[0]
							#t_coeff=recv['t(s)'][0]
							#print "new coeff!"
						except KeyError:
							last_signal=recv['signal'].values[0]
							t=recv['t(s)'].values[0]
							#t_test=t
							#t_coeff=
							#print "last signal !"
						except Exception as e:
							#time.sleep(0.0005)
							#print e
							pass
				first=False
				t_2=time.time()
				t_recv=max(t_recv,t_2-last_t)
				#if t_test!=0 and t_coeff!=0:
					#delta+=t_coeff-t_test
					#if i%100==0:
						#print "delta t signal : ", delta/i
					#i+=1
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
						self.last_coeff=self.coeff
					t1=t2
				#if new_coeff==self.coeff: # if coeff=new_coeff, update the last_coeff
					#self.last_coeff=self.coeff
				#a4=time.time()
				#print "eval coeff time :", (a4-a3)
				Array=pd.DataFrame([[t,last_signal*self.coeff]],columns=self.labels)
				t_3=time.time()
				t_calc=max(t_calc,t_3-t_2)
				try:
					for j in range(len(self.outputs)):
						self.outputs[j].send(Array)
				except:
					pass
				a5=time.time()
				t_send=max(t_send,a5-t_3)
				t_total=max(t_total,a5-t_1)
				if k%100==0:
					print "sleep,recv,calc,send,total: ", t_sleep,t_recv,t_calc,t_send,t_total
					print count*1./k
					t_sleep=0
					t_recv=0
					t_calc=0
					t_send=0
					t_total=0
				k+=1
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in SignalAdapter : ", e
			raise
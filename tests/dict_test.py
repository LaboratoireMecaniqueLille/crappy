import pandas as pd
import os
import gc
from multiprocessing import Process,Pipe
import time
from collections import OrderedDict
#from ctypes import cdll, CDLL

#import pandas as pd
#import numpy as np

#arr = np.random.randn(100000, 5)

#cdll.LoadLibrary("libc.so.6")
#libc = CDLL("libc.so.6")

#def leak():
    #for i in xrange(10000):
        #libc.malloc_trim(0)
        #df = pd.DataFrame(arr.copy())
        #result = df.xs(1000)
        ## result = df.ix[5000]

#class actuator:
	#def __init__(self):
		#pass
	#def set_cmd(self,val):
		#val+=1
#A=actuator()
#comedi_actuators=[A]
a,b=Pipe()
a2,b2=Pipe()
columns=['a','b','c']
data=[1.,2.,3.]
#Data=dict(zip(columns,data))
Data=OrderedDict(zip(columns,data))

def sender(a):
	print "sender :", os.getpid()
	while True:
		Data=pd.DataFrame([[1.,2.,3.]],columns=['a','b','c'])
		#Data={'a':1.,'b':2.,'c':3.}
		a.send(Data)
		#Data=pd.DataFrame([[10.,20.,30.]],columns=['a','b','c'])
		#a.send(Data)
		
def sender2(a):
	print "sender :", os.getpid()
	while True:
		#Data=pd.DataFrame([[1.,2.,3.]],columns=['a','b','c'])
		#columns=['a','b','c']
		#data=[1.,2.,3.]
		##Data=dict(zip(columns,data))
		#Data=OrderedDict(zip(columns,data))
		a.send(Data)
		time.sleep(1)
		#Data=pd.DataFrame([[10.,20.,30.]],columns=['a','b','c'])
		#a.send(Data)


def receive(b):
	return b.recv()

def main(b):  ### this one cause a memory "leak" !!!!! only when the pipe is on
	try:
		print "receiver :", os.getpid()
		last_cmd=0
		i=0
		#gc.disable()
		t=time.time()
		while True:
			Data=b.recv()
			#Data=receive(b)
			cmd=Data['a'].values[0]
			#cmd=Data['a']
			i+=1
			#for comedi_actuator in comedi_actuators:
				#comedi_actuator.set_cmd(cmd)
			#gc.collect()
			#del Data
			#del cmd
			#libc.malloc_trim(0)
			if i%1000==0:
				t1=time.time()
				print i
				print (t1-t)/1000
				t=t1
	except (Exception,KeyboardInterrupt) as e:
		print "Exception in CommandComedi : ", e
		raise
	
def main2(b,k):
	try:
		print "receiver_2 :", os.getpid()
		last_cmd=0
		i=0
		t=time.time()
		while True:
			Data=b.recv()
			cmd=Data['a']
			print Data
			Data['a']=k
			print Data
			i+=1
			if i%100000==0:
				t1=time.time()
				print i
				print (t1-t)/100000
				t=t1
	except (Exception,KeyboardInterrupt) as e:
		print "Exception in CommandComedi : ", e
		raise
	
try:
	p=Process(target=main2,args=(b,-50))
	q=Process(target=sender2,args=(a,))
	r=Process(target=main2,args=(b2,-100))
	s=Process(target=sender2,args=(a2,))
	p.start()
	q.start()
	r.start()
	s.start()
except (Exception,KeyboardInterrupt) as e:
	print "Exception in main : ", e
	try:
		p.terminate()
		q.terminate()
		#r.terminate()
		#s.terminate()
	except:
		q.terminate()

	#def main(self):
		#print "command comedi! :", os.getpid()
		#try:
			#last_cmd=0
			#while True:
				#Data=self.inputs[0].recv()
				#cmd=Data['signal'].values[0]
				#if cmd!= last_cmd:
					#for comedi_actuator in self.comedi_actuators:
						#comedi_actuator.set_cmd(cmd)
					#last_cmd=cmd
				#else:
					#gc.collect()
		#except (Exception,KeyboardInterrupt) as e:
			#print "Exception in CommandComedi : ", e
			#for comedi_actuator in self.comedi_actuators:
				#comedi_actuator.close()
			#raise


import pandas as pd
import os
import gc
from multiprocessing import Process,Pipe
import time
from collections import OrderedDict

a2,b2=Pipe()
DEBUG=True
		
def sender2(a):
	print "sender :", os.getpid()
	while True:
		columns=['a','b','c']
		data=[1.,2.,3.]
		Data=OrderedDict(zip(columns,data))
		a.send(Data)

def main2(b):
	try:
		print "receiver_2 :", os.getpid()
		last_cmd=0
		i=0
		t=time.time()
		while True:
			Data=b.recv()
			cmd=Data['a']
			i+=1
			if i%100000==0:
				t1=time.time()
				try:
					if DEBUG:
						print "DEBUG"
				except NameError:
					pass
				print i
				print (t1-t)/100000
				t=t1
	except (Exception,KeyboardInterrupt) as e:
		print "Exception in CommandComedi : ", e
		raise
	
try:
	r=Process(target=main2,args=(b2,))
	s=Process(target=sender2,args=(a2,))
	r.start()
	s.start()
except (Exception,KeyboardInterrupt) as e:
	print "Exception in main : ", e
	try:
		r.terminate()
		s.terminate()
		#r.terminate()
		#s.terminate()
	except:
		s.terminate()



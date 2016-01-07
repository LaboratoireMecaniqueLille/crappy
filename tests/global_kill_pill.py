import pandas as pd
import os
import gc
from multiprocessing import Process,Pipe, Queue
import Queue as Queue2# necessary to handle the queue.empty exception

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

class Base(object):
	def main(self):
		pass
	def add_queue(self,q):
		self.q=q
	def start(self):
		try:
			self.proc=Process(target=self.main,args=())
			self.proc.start()
		except (Exception,KeyboardInterrupt) as e:
			print "Exception : ", e # only an exception during start is possible, or a keyboardinterrupt
			self.q.put("kill pill")
			self.proc.terminate()
			#raise #raise the error to the next level for global shutdown


class sender(Base):
	def main(self):
		print "sender :", os.getpid()
		#self.q.put("test")
		pill="init"
		while True:
			try:
				try:
					pill=self.q.get_nowait()
				except Queue2.Empty:
					print "empty"
				#print pill
				if pill =="kill pill":
					raise Exception("kill pill")
				else:
					time.sleep(1)
			except (Exception,KeyboardInterrupt) as e:
				print "Exception in sender : ", e
				self.q.put("kill pill")
				raise # raise the exception : does NOT properly kill the main process because it is in another process, but enough to kill the whole thing

			
		
class sender2(Base):
	def main(self):
		print "sender2 :", os.getpid()
		t0=time.time()
		pill="good pill"
		while True:
			try:
				try:
					pill=self.q.get_nowait()
				except Queue2.Empty:
					print "empty2"
					pass
				if pill =="kill pill":
					raise Exception("kill pill")
				else:
					time.sleep(1)
					if time.time()-t0 >4:
						#self.q.put("kill pill")
						raise Exception("die !!")
			except (Exception,KeyboardInterrupt) as e:
				print "Exception in sender2 : ", e
				self.q.put("kill pill")
				raise



try:
	p=sender()
	p2=sender2()
	q = Queue()
	p.add_queue(q)
	p2.add_queue(q)
	p.start()
	p2.start()
	#r.start()
	#s.start()
except (Exception,KeyboardInterrupt) as e:
	print "Exception in main : ", e
	try:
		p.terminate()
		p2.terminate()
		#r.terminate()
		#s.terminate()
	except:
		p2.terminate()

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


# coding: utf-8
from _meta import MasterBlock
import time
import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError
import os
class Streamer(MasterBlock):
	"""
Send a fake stream of data.
	"""
	def __init__(self,labels=['t(s)','signal']):
		"""
Use it for testing.

Parameters
----------
labels : list of str, default = ['t(s)','signal']
	Output labels.
		"""
		self.labels=labels
		
	def main(self):
            try:
                while True:
                    self.i=0
                    time.sleep(2)
                    for output in self.outputs:
                        output.send(output.name) #OrderedDict(zip( output.name,[time.time()-self.t0,self.i])))
                    self.i+=1     
            except TimeoutError:
                raise
            except AttributeError: #if no outputs
                pass
            except Exception as e:
                print "Exception in streamer (pid:{0}).".format(os.getpid(), e)
            except KeyboardInterrupt:
                pass
            except:
                print "Unexpected exception."

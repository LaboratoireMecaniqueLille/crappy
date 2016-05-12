# coding: utf-8
from _meta import MasterBlock
import time
import os
class Reader(MasterBlock):
	"""
Children class of MasterBlock. Read and print the input Link.
	"""
	def __init__(self,k):
		"""
Create a reader that prints k and the input data in continuous.

Parameters
----------
k : printable
	Some identifier for this particular instance of Reader
		"""
		self.k=k  
		
                
	def main(self):
                try:
                    while True:
                                print "Received by {0}: {1}.".format(input_.name,self.data)
                except Exception as e:
                    print "Exception in reader (pid:{0}): {1}".format(os.getpid(), e)
                except KeyboardInterrupt:
                    pass
                except:
                    print "Unexpected exception."
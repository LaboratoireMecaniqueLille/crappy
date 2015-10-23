from _meta import MasterBlock
import os
import numpy as np
import time
np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd

class Saver(MasterBlock):
	"""Saves data in a file"""
	def __init__(self,log_file):
		"""
Saver(log_file)

Saves data in a file. Be aware that the log file needs to be cleaned before 
starting this function, otherwise it just keep writing a the end of the file.

Parameters
----------
log_file : string
	Path to the log file. If non-existant, will be created.

		"""
		#super(Saver, self).__init__()	
		print "saver!"
		self.log_file=log_file
		if not os.path.exists(os.path.dirname(self.log_file)):
			# check if the directory exists, otherwise create it
			os.makedirs(os.path.dirname(self.log_file))
      
	def main(self):
		first=True
		while True:
			#data=self.inputs[0].recv()
			Data=self.inputs[0].recv()	# recv data
			data=Data.values
			fo=open(self.log_file,"a")		# "a" for appending
			fo.seek(0,2)		#place the "cursor" at the end of the file
			if first:
				legend_=Data.columns
				fo.write(str([legend_[i] for i in range(len(legend_))])+"\n")
				first =False
			data_to_save=str(data)+"\n"
			fo.write(data_to_save)
			fo.close()
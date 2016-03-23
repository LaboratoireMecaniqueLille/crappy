# coding: utf-8
from _meta import MasterBlock
#import time
import matplotlib.pyplot as plt

class CameraDisplayer(MasterBlock):
	"""Simple images displayer. Can be paired with StreamerCamera"""
	def __init__(self):
		print "cameraDisplayer!" 

	def main(self):
		try:
			plt.ion()
			fig=plt.figure()
			ax=fig.add_subplot(111)
			first_loop=True
			while True:
				#print "top loop"
				frame=self.inputs[0].recv()
				if frame != None:
					#print frame[0][0]
					if first_loop:
						im = plt.imshow(frame,cmap='gray')
						first_loop=False
					else:
						im.set_array(frame)
					plt.draw()
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CameraDisplayer : ", e
			plt.close('all')
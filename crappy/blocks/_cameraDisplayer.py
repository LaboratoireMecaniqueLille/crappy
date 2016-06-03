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
				#print frame.shape
				#if frame None:
					#print frame[0][0]
                                im = plt.imshow(frame,cmap='gray')
                                plt.pause(0.001)
                                plt.show()
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CameraDisplayer : ", e
			plt.close('all')
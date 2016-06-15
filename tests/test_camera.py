#from _meta import MasterBlock
import time
import crappy
import pandas as pd
import serial
crappy.blocks._meta.MasterBlock.instances=[]
from serial import SerialException 


	############ Start instances #########
	for i in range(1000): # test 1000 fois
		print "try #", i
		camera=crappy.blocks.StreamerCamera("Ximea",width=4242,height=2830,freq=None,save=False,save_directory="/home/ilyesse/Photos_Ilyesse/")
		
		t0=time.time() 
		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.t0(t0)

		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.start()
		time.sleep(10) # 5 seconds should be enough to start the camera.
		
		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.stop()

	########################################### Stopping objects

except (KeyboardInterrupt) as e:
	print "Exception in main :", e

	for instance in crappy.blocks._meta.MasterBlock.instances:
		try:
			instance.stop()
			print "instance stopped : ", instance
		except:
                    print "exception dans le main"
                    pass
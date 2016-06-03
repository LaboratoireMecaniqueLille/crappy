import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
import numpy as np
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances
#import alerte_jerome

t0=time.time()
if __name__ == '__main__':
	try:
            
	########################################### Creating blocks
		camera=crappy.blocks.StreamerCameraG("Ximea",numdevice=0,freq=10,save=False,save_directory="/media/biaxe/SSD1To/Essais_biface_caoutchouc/cam_pc/",xoffset=0,yoffset=0,width=2048,height=2048)
		display=crappy.blocks.CameraDisplayer()
	########################################### Creating links
		link1=crappy.links.Link()
	########################################### Linking objects
		camera.add_output(link1)
		display.add_input(link1)
	########################################### Starting objects
		t0=time.time()
		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.set_t0(t0)

		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.start()
	########################################### Waiting for execution


	########################################### Stopping objects

	except (Exception,KeyboardInterrupt) as e:
		print "Exception in main :", e
		for instance in crappy.blocks._meta.MasterBlock.instances:
			try:
				instance.stop()
			except:
				pass
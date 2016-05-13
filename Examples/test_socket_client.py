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
            reader1 = crappy.blocks.Reader("Reader1")
            #reader2 = crappy.blocks.Reader("Reader2")
            #reader3 = crappy.blocks.Reader("Reader3")
            #reader4 = crappy.blocks.Reader("Reader4")
            client = crappy.blocks.Client(port=9998, time_sync=True)
            
            link1=crappy.links.Link(name="link1")
            #link2=crappy.links.Link(name="link2")
            #link3=crappy.links.Link(name="link3")
            #link4=crappy.links.Link(name="link4")
            
            reader1.add_input(link1)
            #reader2.add_input(link2)
            #reader3.add_input(link3)
            #reader4.add_input(link4)
            
            client.add_output(link1)
            #client.add_output(link2)
            #client.add_output(link3)
            #client.add_output(link4)
            
            t0=client.t0
            for instance in crappy.blocks._meta.MasterBlock.instances:
                    instance.set_t0(t0)

            for instance in crappy.blocks._meta.MasterBlock.instances:
                    instance.start()
            
	except (Exception,KeyboardInterrupt) as e:
		print "Exception in main :", e
		for instance in crappy.blocks._meta.MasterBlock.instances:
			try:
				instance.stop()
			except:
				pass
		
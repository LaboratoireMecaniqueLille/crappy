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
            streamer1 = crappy.blocks.Streamer()
            #streamer2 = crappy.blocks.Streamer()
            #streamer3 = crappy.blocks.Streamer()
            #streamer4 = crappy.blocks.Streamer()
            server = crappy.blocks.Server(port=9998, time_sync=True)
            
            link1=crappy.links.Link(name="link1")
            #link2=crappy.links.Link(name="link2")
            #link3=crappy.links.Link(name="link3")
            #link4=crappy.links.Link(name="link4")
            
            streamer1.add_output(link1)
            #streamer2.add_output(link2)
            #streamer3.add_output(link3)
            #streamer4.add_output(link4)
            
            server.add_input(link1)
            #server.add_input(link2)
            #server.add_input(link3)
            #server.add_input(link4)
            
            t0=time.time()
            for instance in crappy.blocks._meta.MasterBlock.instances:
                    instance.set_t0(t0)

            for instance in crappy.blocks._meta.MasterBlock.instances:
                    instance.start()
        except Exception as e:
            print "Exception in main :", e
            for instance in crappy.blocks._meta.MasterBlock.instances:
                try:
                    instance.stop()
                except Exception as e:
                    print "ee:", e
                    pass
        except KeyboardInterrupt:
            print 'KeyboardInterrupt'
            pass
		
import time
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

t0 = time.time()
if __name__ == '__main__':
    try:
        streamer1 = crappy2.blocks.Streamer()
        # streamer2 = crappy2.blocks.Streamer()
        # streamer3 = crappy2.blocks.Streamer()
        # streamer4 = crappy2.blocks.Streamer()
        server = crappy2.blocks.Server(port=9998, time_sync=True)

        link1 = crappy2.links.Link(name="link1")
        # link2=crappy2.links.Link(name="link2")
        # link3=crappy2.links.Link(name="link3")
        # link4=crappy2.links.Link(name="link4")

        streamer1.add_output(link1)
        # streamer2.add_output(link2)
        # streamer3.add_output(link3)
        # streamer4.add_output(link4)

        server.add_input(link1)
        # server.add_input(link2)
        # server.add_input(link3)
        # server.add_input(link4)

        t0 = time.time()
        for instance in crappy2.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy2.blocks.MasterBlock.instances:
            instance.start()
    except Exception as e:
        print "Exception in main :", e
        for instance in crappy2.blocks.MasterBlock.instances:
            try:
                instance.stop()
            except Exception as e:
                print "ee:", e
                pass
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'
        pass

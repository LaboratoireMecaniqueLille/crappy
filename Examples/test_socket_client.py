import time
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

t0 = time.time()
if __name__ == '__main__':
    try:
        reader1 = crappy2.blocks.Reader("Reader1")
        # reader2 = crappy2.blocks.Reader("Reader2")
        # reader3 = crappy2.blocks.Reader("Reader3")
        # reader4 = crappy2.blocks.Reader("Reader4")
        client = crappy2.blocks.Client(port=9998, time_sync=True)

        link1 = crappy2.links.Link(name="link1")
        # link2=crappy2.links.Link(name="link2")
        # link3=crappy2.links.Link(name="link3")
        # link4=crappy2.links.Link(name="link4")

        reader1.add_input(link1)
        # reader2.add_input(link2)
        # reader3.add_input(link3)
        # reader4.add_input(link4)

        client.add_output(link1)
        # client.add_output(link2)
        # client.add_output(link3)
        # client.add_output(link4)

        t0 = client.t0
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
                print e
    except KeyboardInterrupt:
        pass

import time
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

t0 = time.time()
if __name__ == '__main__':
    try:

        # Creating blocks
        compacter_extenso = crappy2.blocks.Compacter(100)
        save_extenso = crappy2.blocks.Saver("/home/corentin/Bureau/delete2.txt")
        graph_extenso = crappy2.blocks.Grapher("dynamic", ('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

        extenso = crappy2.blocks.VideoExtenso(camera="ximea", numdevice=0, xoffset=0, yoffset=0, width=2048, height=2048,
                                              white_spot=False, display=True)

        # Creating links

        link1 = crappy2.links.Link()
        link2 = crappy2.links.Link()
        link6 = crappy2.links.Link()

        link3 = crappy2.links.Link()
        link4 = crappy2.links.Link()
        link5 = crappy2.links.Link()

        # Linking objects
        extenso.add_output(link3)
        compacter_extenso.add_input(link3)
        compacter_extenso.add_output(link4)
        compacter_extenso.add_output(link5)

        save_extenso.add_input(link4)

        graph_extenso.add_input(link5)

        # Starting objects
        # print "top :",crappy2.blocks._meta.MasterBlock.instances
        t0 = time.time()
        for instance in crappy2.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy2.blocks.MasterBlock.instances:
            instance.start()

            # Waiting for execution

    # Stopping objects
    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy2.blocks.MasterBlock.instances:
            try:
                instance.stop()
            except Exception as e:
                print e

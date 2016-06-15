import time
import crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

t0 = time.time()
if __name__ == '__main__':
    try:

        # Creating blocks
        compacter_extenso = crappy.blocks.Compacter(100)
        save_extenso = crappy.blocks.Saver("/home/corentin/Bureau/delete2.txt")
        graph_extenso = crappy.blocks.Grapher("dynamic", ('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

        extenso = crappy.blocks.VideoExtenso(camera="ximea", numdevice=0, xoffset=0, yoffset=0, width=2048, height=2048,
                                             white_spot=False, display=True)

        # Creating links

        link1 = crappy.links.Link()
        link2 = crappy.links.Link()
        link6 = crappy.links.Link()

        link3 = crappy.links.Link()
        link4 = crappy.links.Link()
        link5 = crappy.links.Link()

        # Linking objects
        extenso.add_output(link3)
        compacter_extenso.add_input(link3)
        compacter_extenso.add_output(link4)
        compacter_extenso.add_output(link5)

        save_extenso.add_input(link4)

        graph_extenso.add_input(link5)

        # Starting objects
        # print "top :",crappy.blocks._meta.MasterBlock.instances
        t0 = time.time()
        for instance in crappy.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy.blocks.MasterBlock.instances:
            instance.start()

            # Waiting for execution

    # Stopping objects
    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy.blocks.MasterBlock.instances:
            try:
                instance.stop()
            except Exception as e:
                print e

import time
# import matplotlib
# matplotlib.use('Agg')
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

if __name__ == '__main__':
    gain = [10] + [1350 for i in xrange(14)] + [5000, -5000]
    sensor = crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[i for i in xrange(16)], gain=gain,
                                         offset=0)
    labels = ['t(s)', 'Trigg'] + ['T' + str(i) for i in xrange(1, 12)] + ['Tdisc1', 'Tdisc2', 'Fn', 'Ft']
    measurebystep = crappy2.blocks.MeasureByStep(sensor, labels=labels, freq=20)
    compacter = crappy2.blocks.Compacter(2)
    grapher = crappy2.blocks.Grapher(('t(s)', 'T1'), length=120)
    padplot = crappy2.blocks.PadPlot(colormap_range=[20, 200])

    link1 = crappy2.links.Link(name='link to compacter')
    link2 = crappy2.links.Link(name='link to grapher')
    link3 = crappy2.links.Link(name='link to padplot')

    measurebystep.add_output(link1)
    compacter.add_input(link1)

    compacter.add_output(link2)
    compacter.add_output(link3)

    padplot.add_input(link3)
    grapher.add_input(link2)

    try:
        t0 = time.time()
        for instance in crappy2.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy2.blocks.MasterBlock.instances:
            instance.start()

    ########################################### Stopping objects

    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        # for instance in crappy2.blocks._meta.MasterBlock.instances:
        # instance.join()
        for instance in crappy2.blocks._meta.MasterBlock.instances:
            try:
                instance.stop()
                print "instance stopped : ", instance
            except:
                pass

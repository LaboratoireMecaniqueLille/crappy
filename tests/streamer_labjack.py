from time import time, sleep
from crappy2.blocks import MasterBlock, Grapher, Saver, MeasureByStep, Streamer, Compacter, StreamerComedi
from crappy2.sensor import LabJackSensor, ComediSensor
from crappy2.links import Link
import random
from crappy2.technical import DataPicker
import crappy2

import signal
import os

def loop1(scan_rate, scan_per_read):
    MasterBlock.instances = []  # Init masterblock instances

    stream = 1
    try:
        # graph = Grapher(zip(4 * ['time(sec)'], stream.labels, length=0)
        if stream:
            sensor = LabJackSensor(mode="streamer", scan_rate_per_channel=scan_rate, scans_per_read=scan_per_read,
                                   channels=[0])
            stream = Streamer(sensor=sensor)
            # graph = Grapher(zip(['time(sec)'], stream.labels[1:]), length=1)
            # compacter = Compacter(50)

            Link1 = Link()
            # Link_to_compacter = Link()
            # save = Saver()
            stream.add_output(Link1)
            flusher = DataPicker(Link1)
            # save = Saver('/home/francois/Code/Tests_Python/perf_saver.bin')
            # save.add_input(Link1)
            # compacter.add_input(Link1)

            # compacter.add_output(Link_to_compacter)

            # graph.add_input(Link1)

        else:

            sensor = LabJackSensor(mode="single", channels=[0, 1, 2, 3, 4])
            # sensor = ComediSensor(channels=[0, 1])
            stream = MeasureByStep(sensor=sensor, labels=['t(s)', 'CHAN0', 'CHAN1'])
            # stream = StreamerComedi(sensor=sensor, labels=['time(sec)'] + ["AD" + str(nb) for nb in xrange(2)], freq=2000, buffsize=10)
            graph = Grapher(('t(s)', 'CHAN0'), length=0)

            Link1 = Link(name='from measurebystep to compacter')
            Link2 = Link(name='from compacter to grapher')
            stream.add_output(Link1)
            compacter = Compacter(20)
            compacter.add_input(Link1)
            compacter.add_output(Link2)
            graph.add_input(Link2)

        t0 = time()

        for instance in MasterBlock.instances:
            instance.t0 = t0

        for instance in MasterBlock.instances:
            instance.start()

    except KeyboardInterrupt:
        sensor.close()
        for instance in MasterBlock.instances:
            instance.stop()

rate1 = 100000
rate2 = int(rate1 / 10.)
loop1(rate1, rate2)


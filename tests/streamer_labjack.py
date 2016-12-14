from time import time, sleep
from crappy.blocks import MasterBlock, Grapher, Saver, MeasureByStep, Streamer, Compacter, StreamerComedi
from crappy.sensor import LabJackSensor, ComediSensor
from crappy.links import Link
import crappy
import random
from crappy.technical import DataPicker
import crappy

import signal
import os

def loop1(scan_rate, scan_per_read):
    MasterBlock.instances = []  # Init masterblock instances

    stream = 1
    try:
        # graph = Grapher(zip(4 * ['time(sec)'], stream.labels, length=0)
        dico = {'mode': "streamer", 'scan_rate_per_channel': scan_rate, 'scans_per_read': scan_per_read, 'channels': [0, 1]}
        if stream:
            # sensor = crappy.technical.LabJack(sensor=dico)
            sensor = crappy.sensor.LabJackSensor(mode="streamer", gain=1, offset=0)
            # sensor = crappy.sensor.DummySensor(channels=[0, 1])
            stream = Streamer(sensor=sensor, labels=['t', 'toto'])
            # graph = Grapher(zip(['time(sec)'], stream.labels[1:]), length=1)
            # compacter = Compacter(50)
            link1 = Link()
            # Link_to_compacter = Link()
            # save = Saver()
            stream.add_output(link1)
            flusher = DataPicker(link1)
            # save = Saver('/home/francois/Code/Tests_Python/perf_saver.bin')
            # save.add_input(link1)
            # compacter.add_input(link1)

            # compacter.add_output(Link_to_compacter)

            # graph.add_input(link1)

        else:

            # sensor = LabJackSensor(mode="single", channels=[0, 1, 2, 3, 4])
            sensor = ComediSensor(channels=[0, 1])
            # sensor = crappy2.sensor.DummySensor(channels=0)
            stream = MeasureByStep(sensor=sensor, labels=['t(s)', 'CHAN0', 'CHAN1'], freq=1000)
            # stream = StreamerComedi(sensor=sensor, labels=['time(sec)'] + ["AD" + str(nb) for nb in xrange(2)], freq=2000, buffsize=10)
            graph = Grapher(('t(s)', 'CHAN0'), ('t(s)', 'CHAN1'), length=100)

            link1 = Link(name='from measurebystep to compacter')
            link2 = Link(name='from compacter to grapher')
            link3 = Link(name='third link')

            stream.add_output(link1)
            compacter = Compacter(10)
            compacter.add_input(link1)
            compacter.add_output(link2)
            compacter.add_output(link3)
            graph.add_input(link2)
            # graph.add_input(link1)
            save = Saver('/home/francois/Code/Tests_Python/test_saver2.txt')
            save.add_input(link3)
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


import time
from collections import OrderedDict

import crappy2
from crappy2.blocks._meta import MasterBlock


class FakeStreamer(MasterBlock):
    def __init__(self, labels=['t(s)', 'signal0', 'signal1', 'signal2']):
        super(FakeStreamer, self).__init__()
        self.labels = labels
        self.i = 0

    def main(self):
        try:
            while 1:
                time.sleep(0.5)
                for output in self.outputs:
                    output.send(OrderedDict(zip(self.labels,
                                                [time.time() - self.t0, self.i, self.i + 10, self.i + 100])))
                self.i += 1
        except Exception as exception:
            print exception


try:
    streamer = FakeStreamer()
    sensor = crappy2.technical.Acquisition(board_name="LabJack", channels=[0, 1, 2], resolution=12, chan_range=0.01,
                                           mode="single")
    streamer = crappy2.blocks.MeasureByStep(sensor, labels=['t(s)', 'signal0', 'signal1', 'signal2'])
    compacter = crappy2.blocks.Compacter(acquisition_step=2)
    grapher = crappy2.blocks.Grapher("dynamic", ('t(s)', 'signal0'), ('t(s)', 'signal1'), ('t(s)', 'signal2'))

    stream_to_compacter = crappy2.links.Link(name="stream_to_compacter")
    comp_to_graph = crappy2.links.Link(name="comp_to_graph")

    streamer.add_output(stream_to_compacter)
    compacter.add_input(stream_to_compacter)
    compacter.add_output(comp_to_graph)
    grapher.add_input(comp_to_graph)

    t0 = time.time()
    for instance in crappy2.blocks.MasterBlock.instances:
        instance.t0 = t0

    for instance in crappy2.blocks.MasterBlock.instances:
        instance.start()

        # Waiting for execution

        # Stopping objects

        # for instance in crappy2.blocks.MasterBlock.instances:
        # instance.stop()

except (Exception, KeyboardInterrupt) as e:
    print "Exception in main :", e
    for instance in crappy2.blocks.MasterBlock.instances:
        try:
            instance.stop()
            print "instance stopped : ", instance
        except Exception as e:
            print e

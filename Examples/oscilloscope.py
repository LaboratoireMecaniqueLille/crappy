# Simple program to plot data. Use the concerned device.

import crappy

channels = [0]

# If comedi as sensor
# sensor = crappy.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3],
#                                      gain=[1, 1, 1, 1], offset=[0.1723, 0.155, -0.005, 0.005])

# If labjack as sensor
sensor = crappy.technical.LabJack(sensor={'channels': channels, 'gain': 1, 'offset': 0, 'resolution': 1}, verbose=True)

# If openDAQ as sensor
# sensor = crappy.technical.OpenDAQ()

stream = crappy.blocks.MeasureByStep(sensor, freq=None, compacter=1000, verbose=True, labels=['t(s)', str(channels[0])])

graph = crappy.blocks.Grapher(('t(s)', str(channels[0])), length=1)
dash = crappy.blocks.Dashboard(nb_digits=6)

crappy.link(stream, dash)
crappy.link(stream, graph)
crappy.start()

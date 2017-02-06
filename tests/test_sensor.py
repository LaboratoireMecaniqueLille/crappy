import crappy
nb_chans = 1
channels = range(nb_chans)

gain = [56] * nb_chans
offset = [0] * nb_chans
# With fake sensor
# sensor = crappy.sensor.DummySensor(channels=[0, 1])

# With Comedi
# sensor = crappy.sensor.ComediSensor(channels=channels,
#                                          gain=gain,
#                                          offset=offset)

# With Labjack (T7 or UE9)
# sensor = crappy.technical.LabJack(sensor={'channels': channels}, verbose=True)

# With OpenDAQ
sensor = crappy.technical.OpenDAQ(channels=[1, 2], nsamples=20)
measure = crappy.blocks.MeasureByStep(sensor, freq=100, verbose=True,
                                      compacter=100)
# grapher = crappy.blocks.Grapher(('t', 'C0'), ('t', 'C1'), length=100)
dash = crappy.blocks.Dashboard()
# crappy.link(measure, grapher)
crappy.link(measure, dash)
crappy.start()

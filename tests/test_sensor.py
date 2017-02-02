import crappy

fake_sensor = crappy.sensor.DummySensor(channels=[0, 1])
measure = crappy.blocks.MeasureByStep(fake_sensor, labels=['t', 'C0', 'C1'],
                                      freq=10, verbose=True, compacter=10)
grapher = crappy.blocks.Grapher(('t', 'C0'), ('t', 'C1'), length=100)

crappy.link(measure, grapher)
crappy.start()

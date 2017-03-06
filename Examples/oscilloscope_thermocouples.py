"""
Simple example to show how to use labjack devices for thermocouple measures.
"""
import crappy

sensor = crappy.sensor.LabJackSensor(channels=["AIN1"],
                                     mode="thermocouple")

stream = crappy.blocks.MeasureByStep(sensor,
                                     labels=['t(s)', 'T'],
                                     verbose=True)

graph = crappy.blocks.Grapher(('t(s)', 'T'), length=180)  #
dash = crappy.blocks.Dashboard(nb_digits=1)

crappy.link(stream, graph)
crappy.link(stream, dash)
crappy.start()

import time
import numpy as np
import crappy



sensor = crappy.sensor.LabJackSensor(channels=["AIN3"], mode="thermocouple")

stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'T'], freq=800,compacter=20,verbose=True)

graph = crappy.blocks.Grapher(('t(s)', 'T'), length=180)
crappy.link(stream,graph)

crappy.start()

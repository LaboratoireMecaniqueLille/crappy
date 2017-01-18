# This program shows how to use labjack as a thermocouple sensor.
import crappy
sensor = {'channels': ["AIN0"],
          'mode': "thermocouple",
          'resolution': 8
          }

sensor = crappy.technical.LabJack(sensor=sensor, verbose=True)
stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'T'], freq=100, compacter=20, verbose=True)
graph = crappy.blocks.Grapher(('t(s)', 'T'), length=180)
dash = crappy.blocks.Dashboard(nb_digits=3)

crappy.link(stream, graph, name='graph')
crappy.link(stream, dash, name='dash')
crappy.start()

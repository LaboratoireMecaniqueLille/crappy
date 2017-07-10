import crappy

# labels = ['temps_python(s)', 'temps_arduino(ms)', 'mode', 'vitesse', 'random']
# labels = ["current_millis", "effort"]
# arduino = crappy.technical.Arduino(port='/dev/ttyACM0',
#                                    baudrate=250000,
#                                    labels=labels)
# measurebystep = crappy.blocks.MeasureByStep(arduino)
#
# graph = crappy.blocks.Grapher(('current_millis', 'effort'), length=10)
# dash = crappy.blocks.Dashboard()
#
# crappy.link(measurebystep, graph)
# crappy.link(measurebystep, dash, name='dash')
# crappy.start()

arduino = crappy.blocks.IOBlock("Arduino",
                                port='/dev/ttyACM0',
                                baudrate=115200,
                                frames=['minitens'],
                                labels=['mil', 'eff'])

graph = crappy.blocks.Grapher(('mil', 'eff'), length=0)
save = crappy.blocks.Saver('/home/francois/Code/_Projets/minitens/toto.csv')

crappy.link(arduino, graph, condition=crappy.condition.Mean(npoints=10))
crappy.link(arduino, save, condition=crappy.condition.Mean(npoints=10))
crappy.start()

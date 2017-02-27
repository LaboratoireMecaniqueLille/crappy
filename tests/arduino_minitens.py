import crappy

labels = ['temps_python(s)', 'temps_arduino(ms)', 'mode', 'vitesse', 'random']
if __name__ == '__main__':
  arduino = crappy.technical.Arduino(port='/dev/ttyACM0',
                                     baudrate=115200,
                                     monitor=True, program='minitens')
  measurebystep = crappy.blocks.MeasureByStep(arduino, labels=labels)
  graph = crappy.blocks.Grapher(('temps_arduino(ms)', 'random'), length=10)
  dash = crappy.blocks.Dashboard()

  crappy.link(measurebystep, graph)
  crappy.link(measurebystep, dash)
  crappy.start()

import crappy

class names(crappy.links.Condition):
  def __init__(self):
    pass
  def evaluate(self, value):
    value["Effort(N)"] = value.pop("e")
    value["Time(sec)"] = value.pop("m")
    return value

arduino = crappy.blocks.IOBlock("Arduino",
                                # port='/dev/ttyACM0',
                                baudrate=115200,
                                frames=['minitens'])
                                # labels=['mil', 'eff'])

graph = crappy.blocks.Grapher(('Time(sec)', 'Effort(N)'), length=1000)
# save = crappy.blocks.Saver('/home/francois/Code/_Projets/minitens/toto.csv')

crappy.link(arduino, graph, condition=names())
# crappy.link(arduino, save, condition=crappy.condition.Mean(npoints=10))
crappy.start()

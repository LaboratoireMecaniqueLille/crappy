import crappy


# class names(crappy.condition.Mean):
#   def __init__(self, **kwargs):
#     self.npoints = kwargs.pop('npoints', 1)
#     self.labels = kwargs.pop('labels', None)
#
#   def evaluate(self, value):
#     value["Time(sec)"] = value.pop("m") / 1000.
#     value["Effort(N)"] = value.pop("e")
#     value["Sens"] = value.pop("s")
#     value[]
#     return value


arduino = crappy.blocks.IOBlock("Arduino",
                                baudrate=115200,
                                frames=['submit', 'monitor', 'minitens'])

graph = crappy.blocks.Sink()
# graph = crappy.blocks.Grapher(('millis', 'effort'), length=1000)
# save = crappy.blocks.Saver('/home/francois/Code/_Projets/minitens/toto.csv')

crappy.link(arduino, graph) #, condition=names())
# crappy.link(arduino, save)  #, condition=crappy.condition.Mean(npoints=10))
# crappy.prepare()
crappy.start()

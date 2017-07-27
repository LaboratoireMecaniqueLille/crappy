"""
Very simple example that show how to communicate with kollmorgen
variator via crappy.
The IOBlock is currently named "Koll", but it needs proper baptism...
Args:
  data: "position", or "speed", depending on which information to display (
  could be improved to give both information)
  axis: 1,2,3 or 4, or "all", depending on which axis to plot. the 4th is the
  rotary encoder.
  labels: to give fancy names to each axis.
"""
import crappy

koll = crappy.blocks.IOBlock("Koll",
                             data="speed",
                             axis="all",
                             # labels=["t(s)", "1"])
                             labels=['t(s)'] + map(str, range(1, 5)))

graph = crappy.blocks.Grapher(("t(s)", "1"),
                              ("t(s)", "2"),
                              ("t(s)", "3"),
                              ("t(s)", "4"), length=1000)

crappy.link(koll, graph)
crappy.start()

import crappy

opendaq = crappy.technical.OpenDAQ()

measure = crappy.blocks.MeasureByStep(opendaq, labels=['t(s)', 'AN1'])
compact = crappy.blocks.Compacter(10)
grapher = crappy.blocks.Grapher(('t(s)', 'AN1'), length=20)

link = crappy.links.Link()
link2 = crappy.links.Link()
measure.add_output(link)
compact.add_input(link)

compact.add_output(link2)
grapher.add_input(link2)
crappy.start()


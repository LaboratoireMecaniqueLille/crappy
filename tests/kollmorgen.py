import crappy

koll = crappy.blocks.IOBlock("Koll",
                             data="position",
                             axis="all",
                             labels=['t(s)'] + map(str, range(1, 5)))

graph = crappy.blocks.Grapher(("t(s)", '1'),
                              ("t(s)", "2"),
                              ("t(s)", '3'),
                              ("t(s)", '4'), length=1000)

crappy.link(koll, graph)
crappy.start()

import crappy

graph = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
graph2 = crappy.blocks.Grapher(('t(s)', 'x(pix)'), ('t(s)', 'y(pix)'))
gres = crappy.blocks.Grapher(('t(s)','res'))

correl = crappy.blocks.DISCorrel(camera="Webcam",show_image=True,residual=True)

crappy.link(correl,graph)
crappy.link(correl,gres)
crappy.link(correl,graph2)
crappy.start()

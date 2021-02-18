import crappy


graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),
                                      length=0)

extenso = crappy.blocks.Video_extenso(camera="Webcam",show_image=True)

crappy.link(extenso,graph_extenso)


gui = crappy.blocks.GUI()
crappy.link(gui,extenso)

crappy.start()

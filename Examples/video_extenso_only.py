import crappy

if __name__ == '__main__':

  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),
                                        length=0)

  extenso = crappy.blocks.VideoExtenso(camera="XimeaCV", compacter=10)

  crappy.link(extenso,graph_extenso)
  crappy.start()

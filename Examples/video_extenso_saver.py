import crappy

if __name__ == '__main__':

  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),
                                        length=0)

  extenso = crappy.blocks.Video_extenso(camera="Webcam",white_spots=False,show_image=True)

  crappy.link(extenso,graph_extenso)
  s = crappy.blocks.Saver("./test/data.csv",delay=1,labels=['t(s)','Exx(%)'])
  crappy.link(extenso,s)
  crappy.start()

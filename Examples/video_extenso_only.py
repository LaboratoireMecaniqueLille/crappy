import time
import crappy

if __name__ == '__main__':

  # Creating blocks
  #save_extenso = crappy.blocks.Saver("DELME/delete2.txt")
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),length=0)

  extenso = crappy.blocks.VideoExtenso(camera="ximea", numdevice=0,
                                xoffset=0, yoffset=0, width=2048, height=2048,
                                white_spot=False, display=True, compacter=10)

  crappy.link(extenso,graph_extenso)
  crappy.start()

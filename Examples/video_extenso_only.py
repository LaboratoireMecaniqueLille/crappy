import time
import crappy

if __name__ == '__main__':

  # Creating blocks
  #save_extenso = crappy.blocks.Saver("DELME/delete2.txt")
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),length=0)

  extenso = crappy.blocks.VideoExtenso(camera="XimeaCV", compacter=10)
            #height=1024,width=1024,xoffset=512,yoffset=512)

  crappy.link(extenso,graph_extenso)
  crappy.start()

#coding: utf-8
from __future__ import print_function
import time
import cv2
import numpy as np

from fields import get_fields,Projector
import crappy


XOFF = 652
YOFF = 920
W = 1240
H = 300

H2 = H

XMIN,XMAX = 130,450
W2 = XMAX-XMIN


class CorrelRT(crappy.blocks.MasterBlock):
  #def __init__(self):
  #  crappy.blocks.MasterBlock.__init__(self)

  def prepare(self):
    self.labels = ['t(s)','x','y','Exx(%)','Eyy(%)']
    self.cam = cv2.VideoCapture(cv2.CAP_XIAPI)
    for i in range(10):
      r,f = self.cam.read()
    self.img0 = np.array(f[YOFF:YOFF+H,XOFF:XOFF+W]) # [:,:,0]
    self.h,self.w = self.img0.shape
    self.fields = get_fields(['x','y','exx','eyy'],H2,W2)
    self.p = Projector(self.fields)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    self.dis.setVariationalRefinementIterations(20)
    self.dis.setVariationalRefinementAlpha(20)
    self.dis.setVariationalRefinementDelta(1)
    self.dis.setVariationalRefinementGamma(0)
    self.dis.setFinestScale(1)

  def begin(self):
    r,f = self.cam.read()
    self.img0 = np.array(f[YOFF:YOFF+H,XOFF:XOFF+W]) # [:,:,0]

  def loop(self):
    r,f = self.cam.read()
    img = np.array(f[YOFF:YOFF+H,XOFF:XOFF+W]) # [:,:,0]
    t = time.time()
    flow = self.dis.calc(self.img0,img,None)
    print("flow shape",flow.shape)
    data = self.p.get_scal(flow[:,XMIN:XMAX,:])
    data[2] *= 200/W2
    data[3] *= 200/H2
    print(data)
    self.send([t-self.t0]+data)

  def end(self):
    self.cam.release()


save_path = "biotens_data/"
timestamp = time.ctime()[:-5].replace(" ","_")
save_path += timestamp+"/"
# Creating F sensor
effort = crappy.blocks.IOBlock("Comedi",channels=[0], gain=[-53.3],labels=['t(s)','F(N)'])
# grapher
graph_effort = crappy.blocks.Grapher(('t(s)','F(N)'))
crappy.link(effort,graph_effort)
# and saver
save_effort = crappy.blocks.Saver(save_path+"effort.csv")
crappy.link(effort,save_effort)
b = crappy.actuator.Biotens()
b.open()
b.reset_position()
b.set_position(5,50)
# Creating biotens technical
biotens = crappy.blocks.Machine([{'type':'biotens','port':'/dev/ttyUSB0','pos_label':'position1','cmd':'cmd'}])  # Used to initialize motor.
#graph_pos= crappy.blocks.Grapher(('t(s)', 'position1'))
#crappy.link(biotens,graph_pos)
# And saver
save_pos= crappy.blocks.Saver(save_path+'position.csv')
crappy.link(biotens,save_pos)

# To pilot the biotens
signal_generator = crappy.blocks.Generator([
  {'type':'cyclic','condition1':'Exx(%)>20','value1':5,
    'condition2':'Exx(%)<5','value2':-5,'cycles':1000}
  ],freq=100)
crappy.link(effort,signal_generator)
crappy.link(signal_generator,biotens)

# Correl RT
correl = CorrelRT()
crappy.link(correl,signal_generator)

# Saver
save_correl = crappy.blocks.Saver(save_path+'extenso.csv',labels=['t(s)','Exx(%)','Eyy(%)'])
crappy.link(correl, save_correl)
# And grapher
graph_correl = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
crappy.link(correl, graph_correl)

graph_xy = crappy.blocks.Grapher(('t(s)','x'),('t(s)','y'))
crappy.link(correl,graph_xy)

#And here we go !
crappy.prepare()
input("READY???")
print("GO!")
crappy.launch()

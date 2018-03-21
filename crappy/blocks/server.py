# coding: utf-8

from __future__ import print_function

import socket
import pickle

from .masterblock import MasterBlock

class Server(MasterBlock):
  def __init__(self,port=1148,nclient=1,header=4,bs=4096,delay=1):
    """
    """
    MasterBlock.__init__(self)
    self.niceness = -10
    self.port = port
    self.nclient = nclient
    self.client = []
    self.header = header
    self.bs = bs
    self.delay = delay

  def prepare(self):
    self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    self.socket.bind(('',self.port))
    self.socket.listen(self.nclient)
    while len(self.client) < self.nclient:
      conn,addr = self.socket.accept()
      self.client.append(conn)
      print("New client:",addr,
          "(%d/%d)"%(len(self.client),self.nclient))

  def loop(self):
    data = self.inputs[0].recv_delay(self.delay)
    s = pickle.dumps(data)
    h = []
    nbytes = len(s)
    for i in range(self.header):
      h.append(nbytes%256)
      nbytes = (nbytes - h[-1])//256
    if nbytes:
      raise EOFError("header cannot encode this size "+str(nbytes))
    s = "".join([chr(c) for c in h])+s
    for c in self.client:
      c.send(s)

  def finish(self):
    for c in self.client:
      try:
        c.close()
      except:
        pass
    try:
      self.socket.close()
    except:
      pass

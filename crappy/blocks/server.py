# coding: utf-8

from __future__ import print_function

import socket

from .masterblock import MasterBlock

class Server(MasterBlock):
  def __init__(self,port=1148,nclient=1,header=(b'\x05\x01\x02\x01',4),
      bs=4096,delay=1,dump_method='pickle'):
    """
    """
    MasterBlock.__init__(self)
    self.niceness = -10
    self.port = port
    self.nclient = nclient
    self.client = []
    self.header,self.header_len = header
    self.bs = bs
    self.delay = delay
    if dump_method == 'pickle':
      import pickle
      self.dump = pickle.dumps
    elif dump_method == 'json':
      import json
      self.dump = lambda o: json.dumps(o).encode('ascii')
    else:
      self.dump = dump_method

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
    s = self.dump(data)
    h = []
    nbytes = len(s)
    for i in range(self.header_len):
      h.append(nbytes%256)
      nbytes = (nbytes - h[-1])//256
    if nbytes:
      raise EOFError("header cannot encode this size "+str(nbytes))
    s = self.header+b"".join([bytes([c]) for c in h])+s
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

# coding: utf-8

from __future__ import print_function

import socket
import pickle

from .masterblock import MasterBlock

class Client(MasterBlock):
  def __init__(self,address,port=1148,header=4,bs=4096):
    """
    """
    MasterBlock.__init__(self)
    self.niceness = -10
    self.address = address
    self.port = port
    self.header = header
    self.bs = bs

  def prepare(self):
    self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    self.socket.connect((self.address,self.port))

  def decode_size(self,h):
    size = 0
    for i,c in enumerate(h):
      size += 256**i*ord(c)
    return size

  def loop(self):
    s = self.socket.recv(self.bs)
    h,s = s[:self.header],s[self.header:]
    size = self.decode_size(h)
    #print("Expecting",size,"bytes")
    while len(s) < size:
      s += self.socket.recv(min(self.bs,size-len(s)))
    data = pickle.loads(s)
    keys = data.keys()
    #print("K=",keys)
    data = [dict([(k,data[k][i]) for k in keys])
        for i in range(len(data[keys[0]]))]
    for d in data:
      self.send(d)

  def finish(self):
    try:
      self.socket.close()
    except:
      pass

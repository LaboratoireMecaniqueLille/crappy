# coding: utf-8

import socket

from .block import Block


class Server(Block):
  """This block will only start after ``nclients`` are connected.

  The header is a byte sequence to identify the start of a payload. One byte is
  appended to the header: the length of the next field, that will hold `n`, the
  length of the incoming sequence (usually 2 bytes is enough).

  The length of the message is coded in the next `n` bytes and then the message
  is appended.
  """

  def __init__(self, port=1148, nclient=1, header=b'crappy_h\x01\x02\x03',
               bs=4096, delay=.1, dump_method='pickle'):

    Block.__init__(self)
    self.niceness = -10
    self.port = port
    self.nclient = nclient
    self.client = []
    self.header = header
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
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(('', self.port))
    self.socket.listen(self.nclient)
    while len(self.client) < self.nclient:
      conn, addr = self.socket.accept()
      self.client.append(conn)
      print("New client:", addr,
          "(%d/%d)" % (len(self.client), self.nclient))

  def loop(self):
    data = self.inputs[0].recv_delay(self.delay)
    s = self.dump(data)
    h = []
    nbytes = len(s)
    while nbytes:
      h.append(nbytes % 256)
      nbytes = (nbytes - h[-1]) // 256
    s = self.header+bytes([len(h)]) + b"".join([bytes([c]) for c in h]) + s
    for c in self.client:
      c.send(s)

  def finish(self):
    for c in self.client:
      try:
        c.close()
      except Exception:
        pass
    try:
      self.socket.close()
    except Exception:
      pass

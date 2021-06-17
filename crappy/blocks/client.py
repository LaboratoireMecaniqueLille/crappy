# coding: utf-8

"""More documentation coming soon !"""

import socket

from .block import Block


class Client(Block):
  def __init__(self, address, port=1148, header=b'crappy_h\x01\x02\x03',
               bs=4096, load_method='pickle'):
    Block.__init__(self)
    self.niceness = -10
    self.address = address
    self.port = port
    self.header = header
    self.bs = bs
    if load_method == 'pickle':
      import pickle
      self.load = pickle.loads
    elif load_method == 'json':
      import json
      self.load = lambda s: json.loads(s.decode('ascii'))
    else:
      self.load = load_method
    self.data = b''

  def prepare(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect((self.address, self.port))

  @staticmethod
  def decode_size(h):
    size = 0
    for i, c in enumerate(h):
      size += 256 ** i * c
    return size

  def begin(self):
    # print("==BEGIN==")
    # print("data=", self.data)
    while self.header not in self.data:
      # print("No header found!, getting some more")
      self.data = self.data[-len(self.header) - 1:]
      self.data += self.socket.recv(self.bs)
      # print("data=", self.data)
    self.data = self.data[self.data.find(self.header):]
    # print("Got a header! cropping")
    # print("data=", self.data)
    # print("++END BEGIN++")

  def loop(self):
    # print('==LOOP==')
    while len(self.data) < len(self.header) + 1:
      self.data += self.socket.recv(self.bs)
    if not self.data.startswith(self.header):
      print("WARNING data loss in client block!")
      self.begin()
    h_len = self.data[len(self.header)] + len(self.header) + 1
    h = self.data[len(self.header)+1:h_len]
    s = self.data[h_len:]
    # print("data=", self.data)
    # print("s=", s)
    size = self.decode_size(h)
    # print("Expecting", size, "bytes")
    while len(s) < size:
      # print("Still not full, waiting for more")
      # print("S=", s)
      s += self.socket.recv(self.bs)
    if len(s) > size:
      # print('Whoops, got {} when waiting for {}...'.format(len(s),size))
      s, self.data = s[:size], s[size:]
    else:
      self.data = b''
    # print("OK! let's send it")
    # print("S=", s)
    data = self.load(s)
    keys = data.keys()
    for i in range(len(data[next(iter(keys))])):
      self.send(dict([(k, data[k][i]) for k in keys]))
    # print('++END LOOP++')

  def finish(self):
    try:
      self.socket.close()
    except Exception:
      pass

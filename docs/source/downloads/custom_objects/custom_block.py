# coding: utf-8

import crappy
from select import select
import socket
from time import sleep
from struct import pack, unpack, calcsize
import logging


class CustomBlock(crappy.blocks.Block):

  def __init__(self,
               label_in=None,
               label_out=None,
               address_out='localhost',
               port_out=50001,
               address_in='localhost',
               port_in=50001,
               freq=30,
               display_freq=False,
               debug=False):

    super().__init__()

    # Block-level attributes
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Labels in and out
    self._label_in = label_in
    self._label_out = label_out

    # Ports and addresses
    self._address_in = address_in
    self._address_out = address_out
    self._port_in = port_in
    self._port_out = port_out

    # Sockets to manage
    self._sock_in = None
    self._sock_out = socket.socket()
    self._sock_server = socket.socket()

  def prepare(self):

    # If there are incoming Links, trying to connect to a server socket
    if self.inputs:
      retries = 5
      connected = False

      # Allowing several retries for connection
      while retries and not connected:
        try:
          self._sock_out.connect((self._address_out, self._port_out))
          connected = True
        except ConnectionRefusedError:
          retries -= 1
          self.log(logging.DEBUG, f"Could not connect to port {self._port_out}"
                                  f"at address {self._address_out}, "
                                  f"{retries} retires left")
          sleep(2)

      # Not proceeding if we couldn't connect
      if not connected:
        self.log(logging.ERROR, f"Could not connect to port {self._port_out}"
                                f"at address {self._address_out}, aborting !")
        raise ConnectionRefusedError
      self.log(logging.INFO, f"Connected to port {self._port_out} at address "
                             f"{self._address_out}")

    # If there are output Links, set up a server and wait for connections
    if self.outputs:
      self._sock_server.bind((self._address_in, self._port_in))
      self._sock_server.listen(0)
      self._sock_server.setblocking(False)
      self.log(logging.INFO, f"Set up server socket on port {self._port_in}"
                             f"at address {self._address_in}")

      # Waiting for an incoming connection
      ret, *_ = select([self._sock_server], list(), list(), 10)
      # Not proceeding if no other Block trie to connect
      if not ret:
        self.log(logging.ERROR, f"No connection requested on port "
                                f"{self._port_in} at address "
                                f"{self._address_in}, aborting !")
        raise ConnectionError

      # Accepting one connection
      self._sock_in, _ = self._sock_server.accept()
      self._sock_in.setblocking(False)

      self.log(logging.INFO, f"Accepted one connection request on port "
                             f"{self._port_in} at address {self._address_in}")

  def loop(self):

    # Receiving the data from upstream Blocks
    data = self.recv_last_data()

    # Only processing if the time and input labels are present
    if data and self._label_in in data and 't(s)' in data:
      t = data['t(s)']
      val = data[self._label_in]
      # Packing to have a standard message format
      to_send = pack('<ff', t, val)

      # Sending the message
      self.log(logging.DEBUG, f"Sending {to_send} on {self._port_out} at "
                              f"address {self._address_out}")
      self._sock_out.send(to_send)

    # If this socket is defined, data can be received from a server socket
    if self._sock_in is not None:
      ready, *_ = select([self._sock_in], list(), list(), 0)

      # Only proceeding if there is in-waiting data
      if ready:
        msg_in = self._sock_in.recv(calcsize('<ff'))
        self.log(logging.DEBUG, f"received {msg_in} on {self._port_in} at "
                                f"address {self._address_in}")

        # Unpacking the received data and sending to downstream Blocks
        self.send(dict(zip(('t(s)', self._label_out), unpack('<ff', msg_in))))

  def finish(self):

    # Only closing this socket if it has been defined
    if self._sock_in is not None:
      self._sock_in.close()
      self.log(logging.INFO, f"Closed the socket on port {self._port_in} at "
                             f"address {self._address_in}")

    # Closing these sockets in all cases
    self._sock_out.close()
    self._sock_server.close()
    self.log(logging.INFO, f"Closed the sockets on port {self._port_out} at "
                           f"address {self._address_out}")


if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Sine',
                                  'freq': 0.5,
                                  'amplitude': 2,
                                  'condition': 'delay=10'},),
                                cmd_label='cmd',
                                freq=30)

  send = CustomBlock(label_in='cmd')
  recv = CustomBlock(label_out='recv')

  graph = crappy.blocks.Grapher(('t(s)', 'recv'))

  crappy.link(gen, send)
  crappy.link(recv, graph)

  crappy.start()

# coding: utf-8
from multiprocessing import Pipe, Process


class DataPicker:
  """
  Class to flush data on a link, it continuously read
  the data received on a link
  and return the last on get_data call.
  """

  def __init__(self, pipe_in):
    self.pipe_in = pipe_in
    self.parent, self.child = Pipe()
    self.proc = Process(target=self.start, args=(self.child,))
    self.proc.start()

  def start(self, child):
    try:
      while True:
        data_in = self.pipe_in.recv()
        # print "DATA_IN: ", data_in
        if child.poll():
          data = child.recv()
          if data == "break":
            break
          child.send(data_in)
    except KeyboardInterrupt:
      self.close()

  def get_data(self):
    self.parent.send('ok')
    data = self.parent.recv()
    return data

  def close(self):
    self.parent.send("break")
    # self.proc.join()

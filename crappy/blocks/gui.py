# coding: utf-8

from time import time

from .block import Block
from .._global import CrappyStop
from .._global import OptionalModule

try:
  import tkinter as tk
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")


class GUI(Block):
  """Block to send a signal based on a user input."""

  def __init__(self, freq=50, label='step', spam=False):
    Block.__init__(self)
    self.freq = freq
    self.spam = spam  # Send the values only once or at each loop ?
    self.i = 0  # The value to be sent
    self.abort = False
    if isinstance(label, list):
      self.labels = label
    else:
      self.labels = ['t(s)', label]

  def prepare(self):
    self.root = tk.Tk()
    self.root.title("GUI block")
    self.root.protocol("WM_DELETE_WINDOW", self.end)
    self.label = tk.Label(self.root, text='step: 0')
    self.label.pack()
    self.button = tk.Button(self.root, text='Next step', command=self.callback)
    self.button.pack()
    self.send([0, self.i])

  def loop(self):
    if self.spam:
      self.send([time() - self.t0, self.i])
    if self.abort:
      raise CrappyStop
    self.root.update()

  def end(self):
    self.abort = True

  def callback(self):
    self.i += 1
    self.send([time()-self.t0, self.i])
    self.label.configure(text='step: %d' % self.i)

  def finish(self):
    self.root.destroy()

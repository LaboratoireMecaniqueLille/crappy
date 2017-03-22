#coding; utf-8

from .masterblock import MasterBlock
from ..tool import Correl as Correl_class
from collections import OrderedDict
from time import time
import numpy as np

class Correl(MasterBlock):
  """
    This block uses the Correl class (in crappy/technicals/_correl.py)

    The first argument is the (y,x) resolution of the image, and you must
        specify the fields with fields=(...)
    See the docstring of Correl to have more informations about the
        arguments specific to Correl.
    It will try to identify the deformation parameters for each fields.
    If you use custom fields, you can use labels=(...) to name the data
        sent through the link.
    If no labels are specified, custom fields will be named by their position.
    Note that the reference image is only taken once, when the
        .start() method is called (after dropping the first image).
    IMPORTANT: This block has an extra method: .init()
    It is meant to compile all the necessary kernels and should be done before
    starting all the blocks, but AFTER creating, initializing and LINKING them.
    In short, you simply have to add yourCorrelBlock.init() just before
        starting all the blocks.
    You can omit this but the delay before processing the first images
        can be long enough to fill a link and crash.
        Also, the correl block will not send any value before this init is over.
  """

  def __init__(self, img_size, **kwargs):
    MasterBlock.__init__(self)
    self.ready = False
    self.img_size = img_size
    self.Nfields = kwargs.get("Nfields")
    self.verbose = kwargs.get("verbose", 0)
    if self.Nfields is None:
      try:
        self.Nfields = len(kwargs.get("fields"))
      except TypeError:
        print "Error: Correl needs to know the number of fields at init \
with fields=(.,.) or Nfields=k"
        raise NameError('Missing fields')

    # Creating the tuple of labels (to name the outputs)
    self.labels = ('t',)
    for i in range(self.Nfields):
      # If explicitly named with labels=(...)
      if kwargs.get("labels") is not None:
        self.labels += (kwargs.get("labels")[i],)
      # Else if we got a default field as a string,
      # use this string (ex: fields=('x','y','r','exx','eyy'))
      elif kwargs.get("fields") is not None and \
          isinstance(kwargs.get("fields")[i], str):
        self.labels += (kwargs.get("fields")[i],)
      # Custom field and no label given: name it by its position...
      else:
        self.labels += (str(i),)

    # print "[Correl Block] output labels:",self.labels
    # We don't need to pass these arg to the Correl class
    if kwargs.get("labels") is not None:
      del kwargs["labels"]
    # Handle drop parameter: if True, can drop data
    if kwargs.get("drop") is not None:
      self.drop = kwargs["drop"]
      del kwargs["drop"]
    else:
      self.drop = False
    # Handle res parameters: if true, also return the residual
    if kwargs.get("res") is not None:
      self.res = kwargs["res"]
      del kwargs["res"]
      self.labels += ("res",)
    else:
      self.res = False
    self.kwargs = kwargs

  def prepare(self):
    self.correl = Correl_class(self.img_size, **self.kwargs)

  def main(self):
    nLoops = 100  # For testing: resets the original images every nLoops loop
    t2 = time() - 1
    # Drop the first image
    self.inputs[0].recv()
    # This is the only time the original picture is set, so the residual may
    # increase if lightning vary or large displacements are reached
    data = self.inputs[0].recv()
    self.correl.setOrig(data['frame'].astype(np.float32))
    self.correl.prepare()
    tr1 = tr2 = time()
    while True:
      if self.verbose:
        t1 = time()
        print "[Correl block] processed", nLoops / (t1 - t2), "ips"
        print "[Correl block] Receiving images took", \
          (tr2 - tr1) / (t1 - t2) * 100, "% of the time"
        t2 = t1
        tr1 = 0
        tr2 = 0
      for i in range(nLoops):
        tr1 += time()
        if self.drop:
          data = self.inputs[0].recv_last(blocking=True)
        else:
          data = self.inputs[0].recv()
        tr2 += time()
        self.correl.setImage(data['frame'].astype(np.float32))
        out = [data['t(s)']] + self.correl.getDisp().tolist()
        if self.res:
          out += [self.correl.getRes()]
        Dout = OrderedDict(zip(self.labels, out))
        self.send(Dout)

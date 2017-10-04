#coding: utf-8
from __future__ import print_function

from os import path,makedirs
import numpy as np
import tables

from .masterblock import MasterBlock

class Hdf_saver(MasterBlock):
  """
  To save data efficiently in a hdf5 file

  This block is is meant to save data coming by arrays at a high rate (>1kHz).
  It uses the module tables from pytables. Do not forget to specify the type
  of data to be saved (see atom parameter) to avoid casting the data into
  another type, as this could result in data loss or inefficient saving
  kwargs:
    - node: The name of the node where the data will be saved
        (str, default="table")
    - expected_rows: The number of expected rows in the file
        It is used to optimize the dumping (int, default=10**8)
    - atom: This represent the type of data to be stored in the table.
        It can be given as a tables.Atom instance. Also it can be given as
        an Numpy type or a str.
        (tables.Atom/numpy.dtype/str, default=tables.Int16Atom())
    - label: The key of the dict that contains the array to save.
        (str, default="stream")
    - metadata: A dict containing additional info to save in the hdf5 file.
        You can add any type of info, they will be written when opening the
        file at the node corresponding to its key.
        (dict, default={})
  """
  def __init__(self,filename,**kwargs):
    MasterBlock.__init__(self)
    self.filename = filename
    for arg,default in [("node","table"),
                        ("expected_rows",10**8),
                        ("atom",tables.Int16Atom()),
                        ("label","stream"),
                        ("metadata",{}),
                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg(s) in Hdf_saver: "+str(kwargs)
    if not isinstance(tables.Atom,self.atom):
      self.atom = tables.Atom.from_dtype(np.dtype(self.atom))

  def prepare(self):
    assert self.inputs, "No input connected to the hdf_saver!"
    assert len(self.inputs) == 1,\
        "Cannot link more than one block to a hdf_saver!"
    if not path.exists(path.dirname(self.filename)):
      # Create the folder if it does not exist
      makedirs(path.dirname(self.filename))
    if path.exists(self.filename):
      # If the file already exists, append a number to the name
      print("[hdf_saver] WARNING!",self.filename,"already exists !")
      name,ext = path.splitext(self.filename)
      i = 1
      while path.exists(name+"_%05d"%i+ext):
        i += 1
      self.filename = name+"_%05d"%i+ext
      print("[hdf_saver] Using",self.filename,"instead!")
    self.hfile = tables.open_file(self.filename,"w")
    for name,value in self.metadata.iteritems():
      self.hfile.create_array(self.hfile.root,name,value)

  def begin(self):
    data = self.inputs[0].recv_chunk()
    w = data[self.label][0].shape[1]
    self.array = self.hfile.create_earray(
        self.hfile.root,
        self.node,
        self.atom,
        (0, w),
        expectedrows=self.expected_rows)
    for d in data[self.label]:
      self.array.append(d)

  def loop(self):
    data = self.inputs[0].recv_chunk()
    for d in data[self.label]:
      self.array.append(d)

  def finish(self):
    self.hfile.close()

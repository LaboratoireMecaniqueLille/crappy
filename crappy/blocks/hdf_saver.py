#coding: utf-8
from __future__ import print_function

from os import path,makedirs
import tables

from .masterblock import MasterBlock

class Hdf_saver(MasterBlock):
  def __init__(self,filename,**kwargs):
    MasterBlock.__init__(self)
    self.filename = filename
    for arg,default in [("node","table"),
                        ("expected_rows",10**8),
                        ("atom",tables.Int16Atom()),
                        ("label","stream"),
                        ("flush_size",2**24), # 16 MB of cache
                        ("metadata",{}),
                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg(s) in Hdf_saver: "+str(kwargs)

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
    self.rows_until_flush = self.flush_size//self.atom.size*w
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

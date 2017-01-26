#coding: utf-8
from __future__ import print_function, division

import cv2
import Tkinter as tk
from PIL import ImageTk,Image
import numpy as np
from threading import Thread
from multiprocessing import Queue

from time import sleep

maxW = 640
maxH = 480

def camera_config(camera):
  return Camera_config().config(camera)

class Camera_config(object):
  def __init__(self):
    self.loop = True
    self.scales = {}
    self.scales_last = {}
    self.checks = {}
    self.radios = {}

  def config(self,camera):
    self.camera = camera
    self.create_window()
    self.create_scales(filter(lambda x:type(x.limits)==tuple,
                              camera.settings.values()))
    self.create_radios(filter(lambda x:type(x.limits)==dict,
                              camera.settings.values()))
    self.create_checks(filter(lambda x:type(x.limits)==bool,
                              camera.settings.values()))
    while self.loop:
      self.main_loop()
    self.root.destroy()

  def create_window(self):
    self.root = tk.Tk()
    self.root.protocol("WM_DELETE_WINDOW",self.stop)

    self.img_label = tk.Label()
    self.img_label.pack()

  def stop(self):
    self.loop = False

  def create_scales(self,settings):
    for setting in settings:
      name = setting.name
      if type(setting.limits[0]) is float:
        step = (setting.limits[1]-setting.limits[0])/1000
      else:
        step = 1
      self.scales[name]= tk.Scale(self.root,from_=setting.limits[0],
                      resolution=step,length=maxW-40,
            to=setting.limits[1], label=name,orient='horizontal')
      self.scales[name].set(setting.value)
      self.scales[name].pack()
      self.scales_last[name] = setting.value

  def create_checks(self,settings):
    for setting in settings:
      name = setting.name
      self.checks[name] = tk.IntVar()
      b = tk.Checkbutton(self.root, text=name, variable=self.checks[name])
      if setting.value:
        b.select()
      b.pack(anchor=tk.W)

  def create_radios(self,settings):
    for setting in settings:
      name = setting.name
      self.radios[name] = tk.IntVar()
      tk.Label(text=name+" :").pack(anchor=tk.W)
      for k,v in setting.limits.iteritems():
        r = tk.Radiobutton(self.root, text=k, variable=self.radios[name],
                           value=v)
        if setting.value == v:
          r.select()
        r.pack(anchor=tk.W)

  def convert_image(self,img):
    if img.dtype == np.uint16:
      img = (img//4).astype(np.uint8)
    try:
      height,width = img.shape
    except ValueError:
      height,width,d = img.shape
      if d == 3:
        img = img[:,:,[2,1,0]] # BGR to RGB
    ratio = max(width/maxW,height/maxH)
    if ratio >= 1:
      w = int(width/ratio)
      h = int(height/ratio)
    else:
      w = width
      h = height
    return ImageTk.PhotoImage(Image.fromarray(cv2.resize(img,(w,h))))

  def update_scales(self):
    for name,scale in self.scales.iteritems():
      v = scale.get()
      if v != self.scales_last[name]:
        #print("Setting",name,"to",v)
        #camera.settings[name].value = s.get() Would be equivalent!
        setattr(self.camera,name,v)
        self.scales_last[name] = v
          
        if getattr(self.camera,name) == v:
          scale.configure(fg="black",label=name)
        else:
          scale.configure(fg="red",label=name+" ({})".format(
                                      getattr(self.camera,name)))

  def update_radios(self): 
    for k,v in self.radios.iteritems():
      if getattr(self.camera,k) != v.get():
        #print(k,b.get())
        setattr(self.camera,k,v.get())

  def update_checks(self): 
    for k,b in self.checks.iteritems():
      if getattr(self.camera,k) != bool(b.get()):
        #print(k,b.get())
        setattr(self.camera,k,bool(b.get()))

  def main_loop(self):
    img = self.convert_image(self.camera.get_image())
    self.img_label.configure(image=img)
    self.update_scales()
    self.update_radios()
    self.update_checks()
    self.root.update()

#coding: utf-8
from __future__ import print_function, division

import cv2
import Tkinter as tk
from PIL import ImageTk,Image,ImageDraw
import numpy as np
from time import time

maxW = 640
maxH = 480

histH = 80
histW = maxW

histRate = .5 # how often should we refresh the histogram

def make_histogram(img,**kwargs):
  max_value = kwargs.get("max_value",255)
  h_w,h_h = kwargs.get("size",(640,100))
  if len(img.shape) != 2:
    t = img.dtype
    img = np.mean(img,axis=2).astype(t)
  h = np.histogram(img,bins=np.arange(max_value+1))[0] # Length max_value

  xp = np.arange(max_value,dtype=np.float32)/max_value*h_w
  h2 = np.interp(np.arange(h_w),xp,h)
  h2 = h2/max(h2)*h_h

  out_img = Image.new('L',(h_w,h_h))
  draw = ImageDraw.Draw(out_img)
  draw.rectangle((0,0,h_w,h_h),fill=255)
  for i in range(h_w):
    draw.line((i,h_h,i,h_h-h2[i]),fill=0)
  return ImageTk.PhotoImage(out_img)

def camera_config(camera):
  return Camera_config().config(camera)

class Camera_config(object):
  def __init__(self):
    self.loop = True
    self.scales = {}
    self.scales_last = {}
    self.checks = {}
    self.radios = {}
    self.last_hist_update = time()

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

    self.hist_label = tk.Label()
    self.hist_label.pack()
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
          #scale.configure(fg="red",label=name+" ({})".format(
          scale.configure(label=name+" ({})".format(
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

  def update_histogram(self,img):
    self.hist = make_histogram(img,size=(histW,histH),
                max_value=255 if img.max()<=255 else 1023)#For 10 bits cameras
    self.hist_label.configure(image=self.hist)


  def main_loop(self):
    t,img = self.camera.get_image()
    if time() - self.last_hist_update > histRate:
      self.last_hist_update = time()
      self.update_histogram(img)
    img = self.convert_image(img)
    self.img_label.configure(image=img)
    self.update_scales()
    self.update_radios()
    self.update_checks()
    self.root.update()

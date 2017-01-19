#coding: utf-8
from __future__ import print_function, division

import cv2
import Tkinter as tk
from PIL import ImageTk,Image

maxW = 800
maxH = 600

loop = True

def finish():
  global loop
  loop = False

def convert_image(img):
  height,width = img.shape
  ratio = max(width/maxW,height/maxH)
  global h,w
  if ratio >= 1:
    w = int(width/ratio)
    h = int(height/ratio)
  else:
    w = width
    h = height
  return ImageTk.PhotoImage(Image.fromarray(cv2.resize(img,(w,h))))

def camera_config(camera):
  # Dimension of the image: it will be resized to fit in a window of maxW*maxH
  # while keeping the ratio (or kept as is if smaller)
  width,height = camera.width, camera.height
  img = camera.get_image()
  assert img.shape == (height,width),"""Camera sensor is
{}x{} but got {}x{} image""".format(width,height,img.shape[1],img.shape[0])
  root = tk.Tk()
  root.protocol("WM_DELETE_WINDOW",finish)
  img = convert_image(img)
  img_label = tk.Label(image=img)
  #img_label.image=img
  img_label.pack()
  scales = []
  labels = []
  checks = {}
  for name,setting in camera.settings.iteritems():
    if setting.limits:
      if type(setting.limits[0]) in [int,float]:
        if type(setting.limits[0]) is float:
          step = (setting.limits[1]-setting.limits[0])/1000
        else:
          step = 1
        scales.append(tk.Scale(root,from_=setting.limits[0],resolution=step,
              to=setting.limits[1], label=name,length=w-40,orient='horizontal'))
        scales[-1].set(setting.value)
        scales[-1].pack()
        labels.append(tk.Label(root, text=name))
        labels[-1].pack()
      elif type(setting.limits[0]) is bool:
        checks[name] = tk.IntVar()
        b = tk.Checkbutton(root, text=name, variable=checks[name])
        if setting.value:
          b.select()
        b.pack()
  scales_last = map(lambda a:a.get(),scales)
  root.update()


  while loop:
    img = convert_image(camera.get_image())
    img_label.configure(image=img)
    for i in range(len(scales)):
      name = scales[i]['label']
      v = scales[i].get()
      if v != scales_last[i]:
        #print("Setting",name,"to",v)
        #camera.settings[name].value = s.get() Would be equivalent!
        setattr(camera,name,v)
        scales_last[i] = v
          

    for k,b in checks.iteritems():
      if getattr(camera,k) != bool(b.get()):
        print(k,b.get())
        setattr(camera,k,bool(b.get()))
    root.update()

  root.destroy()
"""
  for k in camera.settings:
    setattr(camera,k,camera.settings[k].default+10)"""

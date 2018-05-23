#coding: utf-8

CONFIG = False #True

import crappy
import SimpleITK as sitk

cam = crappy.camera.Webcam()
cam.open()
if CONFIG:
  crappy.tool.Camera_config(cam).main()
sitk.WriteImage(sitk.GetImageFromArray(cam.get_image()[1]),"photo.tiff")
cam.close()

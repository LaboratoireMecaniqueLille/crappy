# coding: utf-8

"""
Using directly the camera object to take a picture.

Use `CONFIG=True` to open the configuration GUI.

Required hardware:
  - Any camera (replace Webcam by the desired camera if necessary)
"""

import SimpleITK as Sitk

import crappy

if __name__ == "__main__":
  CONFIG = False  # True

  cam = crappy.camera.Webcam()
  cam.open()
  if CONFIG:
    crappy.tool.Camera_config(cam).main()
  Sitk.WriteImage(Sitk.GetImageFromArray(cam.get_image()[1]), "photo.tiff")
  cam.close()

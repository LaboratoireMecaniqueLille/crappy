# coding: utf-8

"""
Very simple program that displays the output of a chosen camera.

Required hardware:
  - Any camera (select the desired camera in the terminal)
"""

import crappy

if __name__ == "__main__":

  # Retrieving the list of available cameras
  cam_list = list(crappy.camera.camera_dict.keys())

  # Asking the user to pick one camera to use for the test
  ret = -1
  for index, cam in enumerate(cam_list):
    print(index, cam)
  while ret not in range(len(cam_list)):
    try:
      ret = int(input("Which camera do you want to use ? "
                      "(CTRL+C to abort) >>> "))
    except (TypeError, ValueError):
      ret = -1
      print("Invalid choice !\n")
    if ret not in range(len(cam_list)):
      ret = -1
      print("Invalid choice !\n")
  cam = cam_list[ret]

  # Instantiating the Camera Block
  camera = crappy.blocks.Camera(camera=cam, display_freq=True,
                                display_images=True, displayer_framerate=20)

  # Starting the test
  crappy.start()

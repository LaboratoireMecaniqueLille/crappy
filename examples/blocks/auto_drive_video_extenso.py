# coding: utf-8

"""
This example demonstrates the use of the AutoDriveVideoExtenso Block. It does
not require any hardware to run.

This Block can drive an Actuator on which a camera performing
video-extensometry is mounted. As the studied sample is stretched, the center
of the spots might move away from the center of the image. The
AutoDriveVideoExtenso Block drives the Actuator so that both centers stay close
together.

In this example, a fake strain is generated on a static image of a sample with
markers. The level of strain is controlled by a Generator Block. A VideoExtenso
Block tracks the markers and outputs their position to the
AutoDriveVideoExtenso Block. This Block normally drives the Actuator based on
this input. Here, instead, the value that normally drives the Actuator (the
difference in pixels between the center of the markers and the center of the
image) is displayed on a Grapher.

After starting this script, just sit back and watch it run. This demo normally
ends automatically after 2 minutes, but can also be stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # Loading the example image for performing video-extensometry
  # On this image, the center of the spots is slightly on the left, so there's
  # a negative difference with respect to the center of the image
  # This image is distributed with Crappy
  img = crappy.resources.ve_markers

  # The Generator Block that drives the fake strain on the image
  # It applies a cyclic strain that makes the spots move away and then closer
  # When stretching, the center of the spots shifts to the left and the
  # difference with the center of the image become even more negative
  gen = crappy.blocks.Generator(
      # Using a CyclicRamp Path to generate cyclic linear stretching
      ({'type': 'CyclicRamp',
        'speed1': 1,  # Stretching at 1%/s
        'speed2': -1,  # Relaxing at 1%/s
        'condition1': 'Exx(%)>20',  # Stretching until 20% strain
        'condition2': 'Exx(%)<0',  # Relaxing until 0% strain
        'cycles': 3,  # The test stops after 3 cycles
        'init_value': 0},),  # Mandatory to give as it's the first Path
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='Exx(%)',  # The generated signal corresponds to a strain

      # Sticking to default for the other arguments
      )

  # The Block that drives the Actuator so that the center of the spots stays in
  # the middle of the image
  # In this example it does not truly influence the generated image, but you
  # can get an idea of how it works
  # In addition to driving the Actuator, this Block outputs the calculated
  # difference between the center of the spots and the center of the image
  # The speed command sent to the Actuator is proportional to this difference
  auto_drive = crappy.blocks.AutoDriveVideoExtenso(
      {'type': 'FakeMotor'},  # FakeMotor so that no hardware is needed
      gain=1,  # The gain to apply to the center difference before sending the
      # speed command to the Actuator
      direction='x+',  # We want to be centered in the x direction, and we
      # suppose that a positive commands shifts the center of the spots to the
      # right of the image
      pixel_range=632,  # The width of the image in pixels
      max_speed=10,  # The maximum speed command to send to the Actuator
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # The Block that calculates the strain of the image by tracking the
  # displacement of the spots
  # This Block is actually also the one that generates the fake strain on the
  # image, but that wouldn't be the case in real life
  # It takes the target strain as an input, and outputs both the computed
  # strain and the positions of the detected spots
  video_extenso = crappy.blocks.VideoExtenso(
      '',  # The name of Camera to open is ignored bc image_generator is given
      config=True,  # Displaying the configuration window before starting
      display_images=True,  # Displaying the image and the detected spots
      # during the test
      freq=50,  # Lowering the default frequency because it's just a demo
      save_images=False,  # We don't want images to be recorded in this demo
      image_generator=crappy.tool.ApplyStrainToImage(img),  # This argument
      # makes the Block generate fake strain on the given image, only useful
      # for demos
      # The labels for sending the calculated strain to downstream Blocks
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)')

      # Sticking to default for the other arguments
      )

  # This Grapher displays the real-time value of the difference between the
  # center of the spots and the center of the image
  # This value is ignored in this demo, but would be the one driving the
  # Actuator in a real-life setup
  graph = crappy.blocks.Grapher(
      ('t(s)', 'diff(pix)')  # Just providing the names of the labels to
      # display in the graph

      # Sticking to default for the other arguments
      )

  # Linking the Blocks together so that each one sends and received the correct
  # information
  # The Generator drives the VideoExtenso, but also takes decision based on its
  # feedback
  crappy.link(video_extenso, auto_drive)
  crappy.link(gen, video_extenso)
  crappy.link(video_extenso, gen)
  crappy.link(auto_drive, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()

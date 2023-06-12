# coding: utf-8

"""
This example demonstrates the use of the Canvas Block. It does not require any
hardware to run.

This Block can display custom text, the current time, colored spots and a color
bar in a window. At each loop, the values of the texts and the color of the
spots are updated based on the values received from the upstream Blocks.

In this example, the Canvas displays the image of a brake pad. A Generator
simulates a temperature variation and sends it to the Canvas. It is displayed
as a spot whose color changes according to the temperature value. The current
time and the current path index of the Generator are also displayed.

After starting this script, just watch how the Canvas evolves. This demo never
ends, and must be stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # Loading the example image for the background of the Canvas
  # This image is distributed with Crappy
  img_path = crappy.resources.paths['pad']

  # This list of dict contains the information to provide about the elements to
  # draw on the Canvas
  # The description of the mandatory and optional keys is given in the
  # documentation of the Canvas
  options = [
    # The first element is a colored dot associated with a text. Each time a
    # value is received over the given label, the displayed value and the color
    # of the dot are updated
    # Here, it displays the output of the Generator
    {'type': 'dot_text',
     'coord': (185, 430),
     'text': 'T(°C) = %.1f',
     'label': 'T(°C)'},
    # The second element is the same as the first one, except it has no dot and
    # simply displays the text
    # Here, it displays the index of the current Generator path
    {'type': 'text',
     'coord': (1000, 1000),
     'text': 'Index = %i',
     'label': 'Index'},
    # The third element simply displays the current time
    {'type': 'time',
     'coord': (80, 1000),
     'text': '',
     'label': ''}]

  # This Generator Block generates a signal for the Canvas to display. It
  # generates a sine wave during 5 seconds, then cyclically repeats it. Along
  # with the value of the sine, it also sends the index of the path.
  gen_temp = crappy.blocks.Generator(
      # Generating a Sine wave of frequency 1/5 Hz oscillating between 150 and
      # 250
      ({'type': 'Sine', 'condition': 'delay=5',
        'freq': 0.2, 'amplitude': 100, 'offset': 200},),
      cmd_label='T(°C)',  # The label carrying the value of the sine
      path_index_label='Index',  # The label carrying the index of the current
      # path
      repeat=True,  # After 5 seconds when the paths ends, repeat it and
      # increment the path index
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # The Canvas Block displays and updates the elements given as arguments
  # Here, the background image is a brake pad and the value to display is
  # a fake temperature supposedly recorded on the surface of the pad
  canvas = crappy.blocks.Canvas(
      image_path=img_path,  # The path to the background image to use
      draw=options,  # The list of elements to draw on the Canvas
      color_range=(150, 250),  # The range of the color bar for dot text
      # elements
      title="Demo Canvas",  # The title of the Canvas windows
      freq=10,  # Already a quite high value for this Block

      # Sticking to default for the other arguments
      )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen_temp, canvas)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()

"""
This example shows how to use arduino inside crappy.
The first program ever made which uses arduino is minitens.

To integrate the arduino as a sensor inside crappy, the .ino program should
send a python dictionary formatted string to the serial port. The library
used to do so is ArduinoJson.h.

The arduino crappy class is also able to write to the serial port, to pass
arguments. It should be formatted as a python dictionary.
"""

import crappy

save_string = "/home/astruc/These/Code_python/Minitens/test_poubelle"
save = True
labels = ["current_millis", "effort", "mode", "vitesse", "boucle"]
# This is the name of labels to use,
# picked from the ones returned by the arduino If not defined, everything
# returned from the arduino will be printed. Available:
# - "mode": 0, 1, 2 or 3
# - "vitesse": 0..255
# - "boucle": 0..infinity(?)

#----------EXTENSOMETRIE
#extenso = crappy.blocks.VideoExtenso(camera="XimeaCV", white_spot=False, display=True)
#save_extenso = crappy.blocks.Saver(save_string + '_extenso.csv')
##graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
#graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'),length=0, window_pos=(1920, 1080), window_size=(4, 4))
#crappy.link(extenso, save_extenso)
#crappy.link(extenso, graph_extenso)

# -----------ARDUINO
#frames = ["minitens", "monitor", "submit"]
frames = ["minitens", "submit"]
# These are options to compose the GUI:
# - "monitor" shows the serial read from arduino
# - "submit" adds a field to write to arduino
# - "minitens" is a more user-friendly way to communicate with the minitens
#    software.

class Formatting(crappy.links.Condition):
  def __init__(self):
    """
    Class used to do some operations on the crappy objects.
    """
    pass

  def evaluate(self, value):
    value["arduino_time(s)"] = value["current_millis"] / 1000.
    del value["current_millis"]
    return value


arduino = crappy.technical.Arduino(port='/dev/ttyACM0',
                                   baudrate=115200,
                                   # labels=labels,
                                   frames=frames)

measurebystep = crappy.blocks.MeasureByStep(arduino)

graph = crappy.blocks.Grapher(('arduino_time(s)', 'effort'), length=0, window_pos=(1920, 0), window_size=(4, 4))
dash = crappy.blocks.Dashboard(nb_digits=3, labels=["arduino_time(s)","position","effort"])

crappy.link(measurebystep, graph, name='grapher', condition=Formatting())
crappy.link(measurebystep, dash, condition=Formatting())

if save:
  saver = crappy.blocks.Saver(save_string + '_effort.csv')
  crappy.link(measurebystep, saver, condition=Formatting())
crappy.start()

#OPTIONS GRAPHER
#effort : window_pos = (1920, 0), window_size = (4, 4)
#extenso : window_pos  = (1920, 1080), window_size = (4, 4)

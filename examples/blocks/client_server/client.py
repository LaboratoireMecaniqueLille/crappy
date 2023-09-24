# coding: utf-8

"""
This example demonstrates the use of the ClientServer Block for receiving data
from an MQTT broker. It requires the Mosquitto MQTT broker to be installed on
the computer to run. It also requires to start the server.py script once this
one is running. See https://mosquitto.org/ for installing Mosquitto.

In this example, the ClientServer Block receives data from the MQTT broker, and
sends it to the Grapher and Dashboard Blocks for display. The Grapher displays
the 'value' label against the local and server time.

After starting this script, notice how the curve for the local time starts to
be drawn. This is because of the init_output argument of the ClientServer
Block, that allows sending values before any data is received from the server.
Then, start the server.py script. The server starts sending data, and the
second curve for the server time appears. The 'flag' label on the Dashboard is
also updated. You can then stop the server.py script, and finally stop this one
by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Block received data from the server and outputs it to the Grapher and
  # Dashboard Blocks for visualization. It subscribes to the given topics on
  # the MQTT broker, and these topics carry the data. Because spam is True,
  # data is output at each loop even if no value was received from the broker
  client = crappy.blocks.ClientServer(
      broker=True,  # This Block is in charge of managing the startup and
      # termination of the MQTT broker
      address='localhost',  # Running the test locally on the computer
      port=1148,  # The network port to use for communication
      init_output={'t_server': 0, 'flag': 0, 'value': 0},  # For each label in
      # topics, the initial value to send while no value was received from the
      # server
      topics=('flag', ('t_server', 'value')),  # The topics to subscribe to,
      # carrying the values of the labels to output. Grouped labels are
      # synchronized
      freq=30,  # Lowering the default frequency because it's just a demo
      spam=True,  # The last value of each label in topics is sent at each loop

      # Sticking to default for the other arguments
      )

  # This Grapher Block displays the data of the 'value' label received by the
  # client. It displays it against the server time, and against the local time.
  # The curve for the local time starts displaying right when the test starts
  # as it does not require input from the server, whereas the curve for the
  # server time only appears once the server has started.
  graph = crappy.blocks.Grapher(('t_server', 'value'), ('t(s)', 'value'))

  # This Dashboard Block displays the value of the label 'flag' that is
  # received from the server
  # Its value is 1 when 'value' is positive, and 0 otherwise
  dash = crappy.blocks.Dashboard(
      'flag',  # The name of the label whose value to display
      nb_digits=0,  # The number of decimal digits to display

      # Sticking to default for the other arguments
      )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(client, graph)
  crappy.link(client, dash)

  # Setting no_raise because CTRL+C is the most natural way to stop this demo
  crappy.start(no_raise=True)

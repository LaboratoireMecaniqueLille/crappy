# coding: utf-8

from .block import Block
from .._global import OptionalModule
from typing import Dict, List, Union, Tuple, Any
import time
import subprocess
from threading import Thread
import queue
import ast
import pickle
import socket

try:
  import paho.mqtt.client as mqtt
except (ModuleNotFoundError, ImportError):
  mqtt = OptionalModule("paho.mqtt.client")


class Client_server(Block):
  """Block for exchanging data on a local network using the MQTT protocol.

  This block can send data to an MQTT broker, receive data from this broker by
  subscribing to its topics, and also launch the Mosquitto broker.
  """

  def __init__(self,
               broker: bool = False,
               address: Any = 'localhost',
               port: int = 1148,
               init_output: Dict[str, Any] = None,
               topics: List[Union[str, Tuple[str, str]]] = None,
               cmd_labels: List[Union[str, Tuple[str, str]]] = None,
               labels_to_send:
               List[Union[str, Tuple[str, str]]] = None) -> None:
    """Checks arguments validity and sets the instance attributes.

    Args:
      broker (:obj:`bool`, optional): If :obj:`True`, starts the Mosquitto 
        broker during the prepare loop and stops it during the finish loop. If
        Mosquitto is not installed a :exc:`FileNotFoundError` is raised.
      address (optional): The network address on which the MQTT broker is
        running.
      port (:obj:`int`, optional): A network port on which the MQTT broker is
        listening.
      init_output (:obj:`dict`, optional): A :obj:`dict` containing for each
        label in ``topics`` the first value to be sent in the output link. Must
        be given if ``topics`` is not :obj:`None`.
      topics (:obj:`list`, optional): A :obj:`list` of :obj:`str` and/or 
        :obj:`tuple` of :obj:`str`. Each string corresponds to the name of a 
        crappy label to be received from the broker. Each element of the list 
        is considered to be the name of an MQTT topic, to which the client 
        subscribes. After a message has been received on that topic, the block 
        returns for each label in the topic (i.e. each string in the tuple) the 
        corresponding data from the message.
      cmd_labels (:obj:`list`, optional): A :obj:`list` of :obj:`str` and/or 
        :obj:`tuple` of :obj:`str`. Each string corresponds to the name of a 
        crappy label to send to the broker. Each element of the list is 
        considered to be the name of an MQTT topic, in which the client 
        publishes. Grouping labels in a same topic (i.e. strings in a same 
        tuple) allows to keep the synchronization between signals coming from a 
        same block, as they will be published together in a same message. This 
        is mostly useful for sending a signal along with its timeframe.
      labels_to_send (:obj:`list`, optional): A :obj:`list` of :obj:`str` 
        and/or :obj:`tuple` of :obj:`str`. Allows to rename the labels before 
        publishing data. The structure of ``labels_to_send`` should be the 
        exact same as ``cmd_labels``, with each label in ``labels_to_send`` 
        replacing the corresponding one in ``cmd_labels``. This is especially 
        useful for transferring several signals along with their timestamps, as 
        the label ``'t(s)'`` should not appear more than once in the topics 
        list of the receiving block.

    Note:
      - ``broker``:
        In order for the block to run, an MQTT broker must be running at the
        specified address on the specified port. If not, an
        :exc:`ConnectionRefusedError` is raised. The broker can be started and
        stopped manually by the user independently of the execution of crappy.
        It also doesn't need to be Mosquitto, any other MQTT broker can be
        used.

      - ``topics``:
        The presence of the same label in multiple topics will most likely lead 
        to a data loss.

      - ``cmd_labels``:
        It is not possible to group signals coming from different blocks in a
        same topic.

      - ``labels_to_send``:
        Differences in the structure of ``labels_to_send`` and ``cmd_labels``
        will not always raise an error, but may lead to a data loss.

      - **Single-value tuples**:
        Single-value tuples can be shortened as strings.
        ::

          topics=[('cmd1',), ('cmd2',)]
          cmd_labels=[('cmd1',), ('cmd2',)]
          labels_to_send=[('cmd1',), ('cmd2',)]

        is equivalent to
        ::

          topics=['cmd1', 'cmd2']
          cmd_labels=['cmd1', 'cmd2']
          labels_to_send=['cmd1', 'cmd2']


    Examples:
      - ``topics``:
        If
        ::

          topics=[('t1', 'cmd1'), 'sign']

        the client will subscribe to the topics
        ::

          ('t1', 'cmd1')
          ('sign',)

        The block will return data associated with the labels
        ::

          't1', 'cmd1'
          'sign'


      - ``cmd_labels``:
        If
        ::

          cmd_labels=[('t1', 'cmd1'), 'sign']

        the client will publish data in the form of
        ::

          [[t1_0, cmd1_0], [t1_1, cmd1_1], ...]
          [[sign_0], [sign_1], ...]

        in the topics
        ::

           ('t1', 'cmd1')
           ('sign',)


      - ``labels_to_send``:
        If
        ::

          cmd_labels=[('t(s)', 'cmd'), 'sign']
          labels_to_send=[('t1', 'cmd1'), 'sign']

        the data from labels
        ::

          't(s)', 'cmd'

        will be published in the topic
        ::

          ('t1', 'cmd1')

        and the data from label
        ::

          'sign'

        in the topic
        ::

          ('sign',)

    """

    Block.__init__(self)
    self.niceness = -10
    self.stop_mosquitto = False
    self.broker = broker
    self.address = address
    self.port = port
    self.init_output = init_output
    self.topics = topics
    self.cmd_labels = cmd_labels
    self.labels_to_send = labels_to_send
    self.reader = Thread(target=self._output_reader)
    self.client = mqtt.Client(str(time.time()))

    self.client.on_connect = self._on_connect
    self.client.on_message = self._on_message
    self.client.reconnect_delay_set(max_delay=10)

    if self.topics is None and self.cmd_labels is None:
      print("[Client_server] WARNING: client-server block is neither an input "
            "nor an output !")

    if self.topics is not None:
      self.last_out_val = init_output

  def prepare(self) -> None:
    """Reorganizes the labels lists, starts the broker and connects to it."""

    # Preparing for receiving data
    if self.topics is not None:
      assert self.outputs, "topics are specified but there's no output link "
      self.topics = [(topic,) if not isinstance(topic, tuple) else topic for
                     topic in self.topics]

      # The buffer for received data is a dictionary of queues
      self.buffer_output = {topic: queue.Queue() for topic in self.topics}

      # Placing the initial values in the queues
      assert self.init_output, "init_output values should be provided"
      for topic in self.buffer_output:
        try:
          self.buffer_output[topic].put_nowait([self.init_output[label]
                                                for label in topic])
        except KeyError:
          print("init_output values should be provided for each label")
          raise

    # Preparing for publishing data
    if self.cmd_labels is not None:
      assert self.inputs, "cmd_labels are specified but there's no input link "
      self.cmd_labels = [(topic,) if not isinstance(topic, tuple) else topic
                         for topic in self.cmd_labels]
      if self.labels_to_send is not None:
        for i, topic in enumerate(self.labels_to_send):
          if not isinstance(topic, tuple):
            self.labels_to_send[i] = (self.labels_to_send[i],)

        # Preparing to rename labels to send using a dictionary
        assert len(self.labels_to_send) == len(
          self.cmd_labels), "Either a label_to_send should be given for " \
                            "every cmd_label, or none should be given "
        self.labels_to_send = {cmd_label: label_to_send for
                               cmd_label, label_to_send in
                               zip(self.cmd_labels, self.labels_to_send)}

    # Starting the broker
    if self.broker:
      self._launch_mosquitto()
      self.reader.start()
      time.sleep(5)
      print('[Client_server] Waiting for Mosquitto to start')
      time.sleep(5)

    # Connecting to the broker
    try_count = 15
    while True:
      try:
        self.client.connect(self.address, port=self.port, keepalive=10)
        break
      except socket.timeout:
        print("[Client_server] Impossible to reach the given address, "
              "aborting")
        raise
      except socket.gaierror:
        print("[Client_server] Invalid address given, please check the "
              "spelling")
        raise
      except ConnectionRefusedError:
        try_count -= 1
        if try_count == 0:
          print("[Client_server] Connection refused, the broker may not be "
                "running or you may not have the rights to connect")
          raise
        time.sleep(1)

    self.client.loop_start()

  def loop(self) -> None:
    """Receives data from the broker and/or sends data to the broker.

    The received data is then sent to the crappy blocks connected to this one.
    """

    """Loop for receiving data
    Each queue in the buffer is checked once: if not empty then the first list 
    of data is popped. This data is then associated to the corresponding 
    labels in dict_out. dict_out is finally returned if not empty. All the 
    labels should be returned at each loop iteration, so a buffer stores the 
    last value for each label and returns it if no other value was received."""
    if self.topics is not None:
      dict_out = {}
      for topic in self.buffer_output:
        if not self.buffer_output[topic].empty():
          try:
            data_list = self.buffer_output[topic].get_nowait()
            for label, data in zip(topic, data_list):
              dict_out[label] = data
          except queue.Empty:
            pass
      # Updating the last_out_val buffer, and completing dict_out before
      # sending data if necessary
      if dict_out:
        for topic in self.buffer_output:
          for label in topic:
            if label not in dict_out:
              dict_out[label] = self.last_out_val[label]
            else:
              self.last_out_val[label] = dict_out[label]
        self.send(dict_out)

    """Loop for sending data
    Data is first received as a list of dictionaries. For each topic, trying to 
    find a dictionary containing all the corresponding labels. Once this 
    dictionary has been found, its data is published as a list of list of 
    values."""
    if self.cmd_labels is not None:
      received_data = [link.recv_chunk() if link.poll() else {} for link
                       in self.inputs]
      for topic in self.cmd_labels:
        for dic in received_data:
          if all(label in dic for label in topic):
            if self.labels_to_send is not None:
              self.client.publish(
                str(self.labels_to_send[topic]),
                payload=pickle.dumps(
                  [dic[label] for label in topic]), qos=0)
            else:
              self.client.publish(str(topic),
                                  payload=pickle.dumps(
                                    [dic[label] for label in
                                     topic]),
                                  qos=0)
            break

  def finish(self) -> None:
    """Disconnects from the broker and stops it."""

    # Disconnecting from the broker
    self.client.loop_stop()
    self.client.disconnect()

    # Stopping the broker
    if self.broker:
      try:
        self.proc.terminate()
        self.proc.wait(timeout=15)
        print('[Client_server] Mosquitto terminated with return code',
              self.proc.returncode)
        self.stop_mosquitto = True
        self.reader.join()
      except subprocess.TimeoutExpired:
        print('[Client_server] Subprocess did not terminate in time')
        self.proc.kill()

  def _launch_mosquitto(self) -> None:
    """Starts Mosquitto in a subprocess."""

    try:
      self.proc = subprocess.Popen(['mosquitto', '-p', str(self.port)],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    except FileNotFoundError:
      print("[Client_server] Mosquitto is not installed !")
      raise

  def _output_reader(self) -> None:
    """Reads the output strings from Mosquitto's subprocess."""

    while not self.stop_mosquitto:
      for line in iter(self.proc.stdout.readline, b''):
        print('[Mosquitto] {0}'.format(line.decode('utf-8')), end='')
        if 'Error: Address already in use' in line.decode('utf-8'):
          print('Mosquitto is already running on this port')
      time.sleep(0.1)

  def _on_message(self, client, userdata, message) -> None:
    """Buffers the received data.

    The received message consists in a list of lists of values. Data is placed
    in the right buffer according to the topic, in the form of lists of values.
    """

    try:
      for data_points in zip(*pickle.loads(message.payload)):
        self.buffer_output[ast.literal_eval(message.topic)].put_nowait(
          list(data_points))
    except pickle.UnpicklingError:
      print("[Client_server] Warning ! Message raised UnpicklingError, "
            "ignoring it")

  def _on_connect(self, client, userdata, flags, rc: Any) -> None:
    """Automatically subscribes to the topics when connecting to the broker."""

    print("[Client_server] Connected with result code " + str(rc))

    # Subscribing on connect, so that it automatically resubscribes when
    # reconnecting after a connection loss
    if self.topics is not None:
      for topic in self.topics:
        self.client.subscribe(topic=str(topic), qos=0)
        print("[Client_server] Subscribed to", topic)

    self.client.loop_start()

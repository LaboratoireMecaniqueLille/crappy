# coding: utf-8

from .block import Block
from .._global import OptionalModule
from typing import Dict, List, Union, Tuple, Any, Optional
from time import time, sleep
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
from threading import Thread
from queue import Queue, Empty
from ast import literal_eval
from pickle import loads, dumps, UnpicklingError
from socket import timeout, gaierror
from itertools import chain

try:
  import paho.mqtt.client as mqtt
except (ModuleNotFoundError, ImportError):
  mqtt = OptionalModule("paho.mqtt.client")

topics_type = List[Union[str, Tuple[str, ...]]]


class Client_server(Block):
  """Block for exchanging data on a local network using the MQTT protocol.

  This block can send data to an MQTT broker, receive data from this broker by
  subscribing to its topics, and also launch the Mosquitto broker.
  """

  def __init__(self,
               broker: bool = False,
               address: Any = 'localhost',
               port: int = 1883,
               init_output: Optional[Dict[str, Any]] = None,
               topics: Optional[topics_type] = None,
               cmd_labels: Optional[topics_type] = None,
               labels_to_send: Optional[topics_type] = None,
               verbose: bool = False,
               freq: float = 200,
               spam: bool = False) -> None:
    """Checks arguments validity and sets the instance attributes.

    Args:
      broker: If :obj:`True`, starts the Mosquitto broker during the prepare
        loop and stops it during the finish loop. If Mosquitto is not installed
        a :exc:`FileNotFoundError` is raised.
      address (optional): The network address on which the MQTT broker is
        running.
      port (:obj:`int`, optional): A network port on which the MQTT broker is
        listening.
      init_output (:obj:`dict`, optional): A :obj:`dict` containing for labels
        in ``topics`` the first value to be sent in the output links. Should be
        given in case the data comes from several sources and data for all
        labels may not be available during the first loops.
      topics (:obj:`list`, optional): A :obj:`list` of :obj:`str` and/or 
        :obj:`tuple` of :obj:`str`. Each string corresponds to the name of a 
        crappy label to be received from the broker. Each element of the list 
        is considered to be the name of an MQTT topic, to which the client 
        subscribes. After a message has been received on that topic, the block 
        returns for each label in the topic (i.e. each string in the tuple) the 
        corresponding data from the message. It also returns the current
        timestamp in the label `'t(s)'`.
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
      verbose: If :obj:`True`, displays the looping frequency of the block.
      freq: The block will try to loop at this frequency.
      spam: If :obj:`True`, sends the last received values at each loop even if
        no new values were received from the broker.

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
    self.verbose = verbose
    self.freq = freq

    # Setting the args
    self._broker = broker
    self._address = address
    self._port = port
    self._spam = spam
    self._init_output = init_output if init_output is not None else dict()
    self._reader = Thread(target=self._output_reader)

    self._stop_mosquitto = False

    # Instantiating the client
    self._client = mqtt.Client(str(time()))
    self._client.on_connect = self._on_connect
    self._client.on_message = self._on_message
    self._client.reconnect_delay_set(max_delay=10)

    if topics is None and cmd_labels is None:
      print("[Client_server] WARNING: client-server block is neither an input "
            "nor an output !")

    # Preparing for receiving data
    if topics is not None:
      # Replacing strings with tuples
      self._topics = [topic if isinstance(topic, tuple) else (topic,) for
                      topic in topics]

      # The last out vals are given for each label, not each topic
      self._last_out_val = {label: None for label in chain(*self._topics)}
      print(self._last_out_val)

      # The buffer for received data is a dictionary of queues
      self._buffer_output = {topic: Queue() for topic in topics}

    else:
      self._topics = None
      self._buffer_output = None

    # Preparing for publishing data
    if cmd_labels is not None:
      # Replacing strings with tuples
      self._cmd_labels = [topic if isinstance(topic, tuple) else (topic,)
                          for topic in cmd_labels]

      if labels_to_send is not None:
        # Replacing strings with tuples
        labels_to_send = [topic if isinstance(topic, tuple) else (topic,)
                          for topic in labels_to_send]

        # Making sure the labels to send have the correct syntax
        if len(labels_to_send) != len(cmd_labels):
          raise ValueError("Either a label_to_send should be given for "
                           "every cmd_label, or none should be given ")

        # Preparing to rename labels to send using a dictionary
        self._labels_to_send = {cmd_label: label_to_send for
                                cmd_label, label_to_send in
                                zip(self._cmd_labels, labels_to_send)}
    else:
      self._cmd_labels = None
      self._labels_to_send = None

  def prepare(self) -> None:
    """Starts the broker and connects to it."""

    # Making sure the necessary inputs and outputs are present
    if self._topics is not None and not self.outputs:
      raise ValueError("topics are specified but there's no output link !")
    if self._cmd_labels is not None and not self.inputs:
      raise ValueError("cmd_labels are specified but there's no input link !")

    # Starting the broker
    if self._broker:
      self._launch_mosquitto()
      self._reader.start()
      sleep(2)
      print('[Client_server] Waiting for Mosquitto to start')
      sleep(2)

    # Connecting to the broker
    try_count = 15
    while True:
      try:
        self._client.connect(self._address, port=self._port, keepalive=10)
        break
      except timeout:
        print("[Client_server] Impossible to reach the given address, "
              "aborting")
        raise
      except gaierror:
        print("[Client_server] Invalid address given, please check the "
              "spelling")
        raise
      except ConnectionRefusedError:
        try_count -= 1
        if try_count == 0:
          print("[Client_server] Connection refused, the broker may not be "
                "running or you may not have the rights to connect")
          raise
        sleep(1)

    self._client.loop_start()

  def loop(self) -> None:
    """Receives data from the broker and/or sends data to the broker.

    The received data is then sent to the crappy blocks connected to this one.
    """

    """Loop for receiving data
    Each queue in the buffer is checked once: if not empty then the first list 
    of data is popped. This data is then associated to the corresponding 
    labels in dict_out. dict_out is finally returned if not empty. All the 
    labels should be returned at each loop iteration, so a buffer stores the 
    last value for each label and returns it if no other value was received. In 
    case no value was received yet for a given label, the user can also provide 
    init values to be sent at the beginning."""
    if self._topics is not None:
      dict_out = {}
      for topic in self._buffer_output:
        if not self._buffer_output[topic].empty():
          try:
            data_list = self._buffer_output[topic].get_nowait()
            for label, data in zip(topic, data_list):
              dict_out[label] = data
          except Empty:
            pass
      # Updating the _last_out_val buffer, and completing dict_out before
      # sending data if necessary
      if dict_out or (self._spam and all(val is not None for val in
                                         self._last_out_val.values())):
        for topic in self._buffer_output:
          for label in topic:
            if label not in dict_out:
              if self._last_out_val[label] is not None:
                dict_out[label] = self._last_out_val[label]
              elif label in self._init_output:
                dict_out[label] = self._init_output[label]
              else:
                raise ValueError(f"No value received for the topic {label} and"
                                 f" no init value given !")
            else:
              self._last_out_val[label] = dict_out[label]
        # Adding the timestamp before sending
        dict_out['t(s)'] = time() - self.t0
        self.send(dict_out)

    """Loop for sending data
    Data is first received as a list of dictionaries. For each topic, trying to 
    find a dictionary containing all the corresponding labels. Once this 
    dictionary has been found, its data is published as a list of list of 
    values."""
    if self._cmd_labels is not None:
      received_data = [link.recv_chunk(blocking=False) for link in self.inputs]
      for topic in self._cmd_labels:
        for dic in received_data:
          if dic is not None and all(label in dic for label in topic):
            self._client.publish(
              topic=str(self._labels_to_send[topic]) if
              self._labels_to_send is not None else str(topic),
              payload=dumps([dic[label] for label in topic]),
              qos=0)
            break

  def finish(self) -> None:
    """Disconnects from the broker and stops it."""

    # Disconnecting from the broker
    self._client.loop_stop()
    self._client.disconnect()

    # Stopping the broker
    if self._broker:
      try:
        self._proc.terminate()
        self._proc.wait(timeout=15)
        print('[Client_server] Mosquitto terminated with return code',
              self._proc.returncode)
        self._stop_mosquitto = True
        self._reader.join()
      except TimeoutExpired:
        print('[Client_server] Subprocess did not terminate in time')
        self._proc.kill()

  def _launch_mosquitto(self) -> None:
    """Starts Mosquitto in a subprocess."""

    try:
      self._proc = Popen(['mosquitto', '-p', str(self._port)],
                         stdout=PIPE,
                         stderr=STDOUT)
    except FileNotFoundError:
      print("[Client_server] Mosquitto is not installed !")
      raise

  def _output_reader(self) -> None:
    """Reads the output strings from Mosquitto's subprocess."""

    while not self._stop_mosquitto:
      for line in iter(self._proc.stdout.readline, b''):
        print('[Mosquitto] {0}'.format(line.decode('utf-8')), end='')
        if 'Error: Address already in use' in line.decode('utf-8'):
          print('Mosquitto is already running on this port')
      sleep(0.1)

  def _on_message(self, _, __, message) -> None:
    """Buffers the received data.

    The received message consists in a list of lists of values. Data is placed
    in the right buffer according to the topic, in the form of lists of values.
    """

    try:
      for data_points in zip(*loads(message.payload)):
        self._buffer_output[literal_eval(message.topic)].put_nowait(
          list(data_points))
    except UnpicklingError:
      print("[Client_server] Warning ! Message raised UnpicklingError, "
            "ignoring it")

  def _on_connect(self, _, __, ___, rc: Any) -> None:
    """Automatically subscribes to the topics when connecting to the broker."""

    print("[Client_server] Connected with result code " + str(rc))

    # Subscribing on connect, so that it automatically resubscribes when
    # reconnecting after a connection loss
    if self._topics is not None:
      for topic in self._topics:
        self._client.subscribe(topic=str(topic), qos=0)
        print("[Client_server] Subscribed to", topic)

    self._client.loop_start()

# coding: utf-8

from typing import Union, Any, Optional
from collections.abc import Iterable
from time import time, sleep
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
from threading import Thread
from queue import Queue, Empty
from ast import literal_eval
from pickle import loads, dumps, UnpicklingError
from socket import timeout, gaierror
from itertools import chain
import logging

from .meta_block import Block
from .._global import OptionalModule

try:
  import paho.mqtt.client as mqtt
except (ModuleNotFoundError, ImportError):
  mqtt = OptionalModule("paho-mqtt")

TopicsType = Iterable[Union[str, Iterable[str]]]


class ClientServer(Block):
  """This Block can exchange data on a local network using the MQTT protocol.
  
  It communicates with an MQTT broker, from which it can receive data by
  subscribing to topics, and to which it can send data by publishing in topics.
  This Block can also manage the execution of a Mosquitto broker, so that the
  broker doesn't need to be manually started before running the Crappy script.

  This Block is intended for communication with remote devices, that may run
  Crappy scripts or other scripts handling data with the correct syntax.
  Potential uses include acquisition from battery-powered devices (e.g.
  microcontrollers) acquiring data in enclosed areas or rotating parts, data
  transfer between machines to delegate processing, communication with other
  programs on a same machine, etc.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Client_server* to *ClientServer*
  """

  def __init__(self,
               broker: bool = False,
               address: str = 'localhost',
               port: int = 1883,
               init_output: Optional[dict[str, Any]] = None,
               topics: Optional[TopicsType] = None,
               cmd_labels: Optional[TopicsType] = None,
               labels_to_send: Optional[TopicsType] = None,
               display_freq: bool = False,
               freq: Optional[float] = 200,
               spam: bool = False,
               debug: Optional[bool] = False) -> None:
    """Checks the validity of the arguments and sets the instance attributes.

    Args:
      broker: If :obj:`True`, starts the Mosquitto broker during the prepare
        loop and stops it during the finish loop. If Mosquitto is not installed
        a :exc:`FileNotFoundError` is raised.
      address: The network address on which the MQTT broker is running, as a
        :obj:`str`.
      port: A network port on which the MQTT broker is listening, as an 
        :obj:`int`.
      init_output: A :obj:`dict` containing for each label in ``topics`` the 
        first value to be sent in the output Links. Should be given in case the 
        data comes from several sources and data for all labels may not be 
        available during the first loops. Must also be given is ``spam`` is set
        to :obj:`True`.
      topics: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        :obj:`str` and/or iterables of :obj:`str`. Each string corresponds to
        the name of a label in Crappy. Each element in the iterable (string or
        iterable of strings) is considered to be the name of an MQTT topic, to
        which the client subscribes. After a message has been received on that
        topic, the Block returns for each label in the topic (just the given
        string or each string in the iterable) the corresponding data from the
        message. It also returns the current timestamp in the label `'t(s)'`.
      cmd_labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        :obj:`str` and/or iterables of :obj:`str`. Each string corresponds to
        the name of a label in Crappy. Each element in the iterable (string or
        iterable of strings) is considered to be the name of an MQTT topic, in
        which the client publishes. Grouping labels in a same topic (i.e.
        strings in a same iterable) allows to keep the synchronization between
        signals coming from a same Block, as they will be published together in
        a same message. This is mostly useful for sending a signal along with
        its timeframe.
      labels_to_send: An iterable (like a :obj:`list` or a :obj:`tuple`)
        containing :obj:`str` and/or iterables of :obj:`str`. Allows to rename
        the labels before publishing data. The structure of ``labels_to_send``
        should be the exact same as ``cmd_labels``, with each label in
        ``labels_to_send`` replacing the corresponding one in ``cmd_labels``.
        This is especially useful for transferring several signals along with
        their timestamps, as the label ``'t(s)'`` should not appear more than
        once in the topics.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.

        .. versionadded:: 1.5.10
      spam: If :obj:`True`, sends the last received values at each loop even if
        no new values were received from the broker. When set to :obj:`True`,
        the ``init_output`` must be provided.

        .. versionadded:: 1.5.10
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0

    Note:
      - ``broker``:
        In order for the Block to run, an MQTT broker must be running at the
        specified address on the specified port. If not, an
        :exc:`ConnectionRefusedError` is raised. The broker can be started and
        stopped manually by the user independently of the execution of Crappy.
        It also doesn't need to be Mosquitto, any other MQTT broker can be
        used.

      - ``topics``:
        The presence of the same label in multiple topics will most likely lead
        to a data loss.

      - ``cmd_labels``:
        It is not possible to group signals coming from different Blocks in a
        same topic.

      - ``labels_to_send``:
        Differences in the structure of ``labels_to_send`` and ``cmd_labels``
        will not always raise an error, but may lead to a data loss.

      - **Single-value iterables**:
        Single-value iterables can be shortened as strings.
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

        The Block will return data associated with the labels
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

    self._client: Optional[mqtt.Client] = None
    self._reader: Optional[Thread] = None
    self._proc: Optional[Popen] = None

    super().__init__()
    self.niceness = -10
    self.display_freq = display_freq
    self.freq = freq
    self.debug = debug

    # Setting the args
    self._broker = broker
    self._address = address
    self._port = port
    self._spam = spam
    self._init_output = init_output if init_output is not None else dict()

    self._stop_mosquitto = False
    
    # These attributes may be set later
    self._topics: Optional[list[tuple[str, ...]]] = None
    self._last_out_val: dict[str, Any] = dict()
    self._buffer_output: Optional[dict[tuple[str, ...], Queue]] = None
    self._cmd_labels: Optional[list[tuple[str, ...]]] = None
    self._labels_to_send: Optional[list[tuple[str, ...]]] = None

    if topics is None and cmd_labels is None:
      self.log(logging.WARNING, "The Client-server Block is neither an input "
                                "nor an output !")

    # Preparing for receiving data
    if topics is not None:
      # Replacing strings with tuples
      self._topics = [(topic,) if isinstance(topic, str) else tuple(topic) for
                      topic in topics]

      # The last out vals are given for each label, not each topic
      self._last_out_val = {label: None for label in chain(*self._topics)}

    # Preparing for publishing data
    if cmd_labels is not None:
      # Replacing strings with tuples
      self._cmd_labels = [(topic,) if isinstance(topic, str) else tuple(topic)
                          for topic in cmd_labels]

      if labels_to_send is not None:
        # Replacing strings with tuples
        labels_to_send = [(topic,) if isinstance(topic, str) else tuple(topic)
                          for topic in labels_to_send]

        # Making sure the labels to send have the correct syntax
        if len(labels_to_send) != len(self._cmd_labels):
          raise ValueError("Either a label_to_send should be given for "
                           "every cmd_label, or none should be given ")

        # Preparing to rename labels to send using a dictionary
        self._labels_to_send = {cmd_label: label_to_send for
                                cmd_label, label_to_send in
                                zip(self._cmd_labels, labels_to_send)}

  def prepare(self) -> None:
    """Starts the broker and connects to it."""

    # Making sure the necessary inputs and outputs are present
    if self._topics is not None and not self.outputs:
      raise ValueError("topics are specified but there's no output link !")
    if self._cmd_labels is not None and not self.inputs:
      raise ValueError("cmd_labels are specified but there's no input link !")

    # Setting the buffer here because Queue objects cannot be set during
    # __init__ in spawn multiprocessing mode
    if self._topics is not None:
      # The buffer for received data is a dictionary of queues
      self._buffer_output = {topic: Queue() for topic in self._topics}

    # Starting the broker
    if self._broker:
      self.log(logging.INFO, f"Starting the Mosquitto broker on port "
                             f"{self._port}")
      self._launch_mosquitto()
      # Creating and starting a Thread reading the stdout of the broker
      self._reader = Thread(target=self._output_reader)
      self._reader.start()
      sleep(2)
      self.log(logging.INFO, "Waiting for Mosquitto to start")
      sleep(2)

    # Instantiating the client here as it cannot be set during __init__ in
    # spawn multiprocessing mode
    self._client = mqtt.Client(str(time()))
    self._client.on_connect = self._on_connect
    self._client.on_message = self._on_message
    self._client.reconnect_delay_set(max_delay=10)

    # Connecting to the broker
    try_count = 15
    while True:
      try:
        self.log(logging.INFO, f"Connecting to the broker at address "
                               f"{self._address} on port {self._port}")
        self._client.connect(self._address, port=self._port, keepalive=10)
        break
      except timeout:
        raise TimeoutError("Impossible to reach the given address, aborting")
      except gaierror:
        raise ValueError("Invalid address given, please check the spelling")
      except ConnectionRefusedError:
        try_count -= 1
        if try_count == 0:
          raise ConnectionRefusedError("Connection refused, the broker may not"
                                       " be running or you may not have the "
                                       "rights to connect")
        sleep(1)

    self.log(logging.INFO, "Starting the client loop")
    self._client.loop_start()

  def loop(self) -> None:
    """Receives data from the broker and/or sends data to the broker.

    The received data is then sent to the crappy Blocks connected to this one.
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
      dict_out = dict()
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
      if dict_out or self._spam:
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
      received_data = self.recv_all_data_raw()
      for topic in self._cmd_labels:
        for dic in received_data:
          if dic is not None and all(label in dic for label in topic):
            if self._labels_to_send is not None:
              topic_to_send = self._labels_to_send[topic]
            else:
              topic_to_send = topic
            self._client.publish(
              topic=str(topic_to_send),
              payload=dumps([dic[label] for label in topic]),
              qos=0)
            self.log(logging.DEBUG, f"Sent {[dic[label] for label in topic]}"
                                    f"on the topic {topic_to_send} with QOS 0")
            break

  def finish(self) -> None:
    """Disconnects from the broker and stops it."""

    # Disconnecting from the broker
    if self._client is not None:
      self.log(logging.INFO, "Stopping the client loop")
      self._client.loop_stop()
      self.log(logging.INFO, "Disconnecting from the broker")
      self._client.disconnect()

    # Stopping the broker
    if self._broker and self._proc is not None:
      try:
        self.log(logging.INFO, "Stopping the Mosquitto broker")
        self._proc.terminate()
        self._proc.wait(timeout=15)
        self.log(logging.INFO, f"Mosquitto terminated with return code "
                               f"{self._proc.returncode}")
        self._stop_mosquitto = True
        if self._reader is not None:
          self._reader.join(0.2)
          if self._reader.is_alive():
            self.log(logging.WARNING, "Reader thread failed to stop !")

      except TimeoutExpired:
        self.log(logging.WARNING, "Mosquitto did not terminate in time, "
                                  "killing it")
        self._proc.kill()

  def _launch_mosquitto(self) -> None:
    """Starts Mosquitto in a subprocess."""

    try:
      self._proc = Popen(['mosquitto', '-p', str(self._port)],
                         stdout=PIPE,
                         stderr=STDOUT)
    except FileNotFoundError:
      raise FileNotFoundError("Mosquitto is not installed !")

  def _output_reader(self) -> None:
    """Reads the output strings from Mosquitto's subprocess."""

    while not self._stop_mosquitto:
      for line in iter(self._proc.stdout.readline, b''):
        self.log(logging.INFO, f"[Mosquitto] {line.decode()}")
        if 'Error: Address already in use' in line.decode():
          self.log(logging.WARNING, "Mosquitto is already running on this "
                                    "port !")
      sleep(0.1)

  def _on_message(self, _, __, message) -> None:
    """Buffers the received data.

    The received message consists in a list of lists of values. Data is placed
    in the right buffer according to the topic, in the form of lists of values.
    """

    try:
      self.log(logging.DEBUG, f"Received message from the broker: {message}")
      for data_points in zip(*loads(message.payload)):
        self._buffer_output[literal_eval(message.topic)].put_nowait(
          list(data_points))
    except UnpicklingError:
      self.log(logging.WARNING, f"Received message caused UnpicklingError, "
                                f"ignoring it: {message}")

  def _on_connect(self, _, __, ___, rc: Any) -> None:
    """Automatically subscribes to the topics when connecting to the broker."""

    self.log(logging.INFO, f"Connected to the broker with return code {rc}")

    # Subscribing on connect, so that it automatically resubscribes when
    # reconnecting after a connection loss
    if self._topics is not None:
      for topic in self._topics:
        self._client.subscribe(topic=str(topic), qos=0)
        self.log(logging.INFO, f"Subscribed to topic {topic}")

    self._client.loop_start()

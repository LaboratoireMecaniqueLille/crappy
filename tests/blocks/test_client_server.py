# coding: utf-8

import logging
from multiprocessing import Value
from pickle import dumps, loads
from queue import Queue
from socket import gaierror, timeout
from subprocess import TimeoutExpired
from typing import Any
from unittest.mock import patch
from crappy._global import OptionalModule
from crappy.blocks.client_server import ClientServer
import crappy.blocks.client_server as client_server_module

from ..block import BlockTestBase, TestBlock, link


class FakeMQTTClient:
  """Small paho.mqtt.client.Client test double."""

  instances: list['FakeMQTTClient'] = list()
  connect_errors: list[Exception] = list()

  def __init__(self, *args, **kwargs) -> None:
    """Stores construction arguments and initializes call records."""

    self.args = args
    self.kwargs = kwargs
    self.client_id = kwargs.get('client_id', args[0] if args else None)

    self.on_connect = None
    self.on_message = None
    self.reconnect_delay_calls = list()
    self.connect_calls = list()
    self.loop_start_calls = 0
    self.loop_stop_calls = 0
    self.disconnect_calls = 0
    self.subscriptions = list()
    self.publications = list()

    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears state shared by all fake clients."""

    cls.instances = list()
    cls.connect_errors = list()

  def reconnect_delay_set(self, max_delay: int) -> None:
    """Records reconnect delay configuration."""

    self.reconnect_delay_calls.append(max_delay)

  def connect(self, address: str, port: int, keepalive: int) -> None:
    """Records connection attempts and raises queued errors if any."""

    self.connect_calls.append((address, port, keepalive))

    if self.connect_errors:
      raise self.connect_errors.pop(0)

  def loop_start(self) -> None:
    """Records that the MQTT network loop was started."""

    self.loop_start_calls += 1

  def loop_stop(self) -> None:
    """Records that the MQTT network loop was stopped."""

    self.loop_stop_calls += 1

  def disconnect(self) -> None:
    """Records that the client was disconnected."""

    self.disconnect_calls += 1

  def subscribe(self, topic: str, qos: int) -> None:
    """Records topic subscriptions."""

    self.subscriptions.append((topic, qos))

  def publish(self, topic: str, payload: bytes, qos: int) -> None:
    """Records publications."""

    self.publications.append((topic, payload, qos))


class FakeMQTTModule:
  """Module-shaped test double exposing a Client constructor."""

  Client = FakeMQTTClient


class FakeMessage:
  """Minimal MQTT message object."""

  def __init__(self, topic: tuple[str, ...], payload: bytes) -> None:
    """Stores the topic string and payload as paho would expose them."""

    self.topic = str(topic)
    self.payload = payload


class FakeProcess:
  """Small subprocess.Popen test double for Mosquitto cleanup."""

  def __init__(self, wait_error: Exception | None = None) -> None:
    """Initializes process state."""

    self.wait_error = wait_error
    self.returncode = 0
    self.terminated = False
    self.killed = False
    self.wait_calls = list()

  def terminate(self) -> None:
    """Records terminate calls."""

    self.terminated = True

  def wait(self, timeout: float) -> None:
    """Records wait calls and raises an optional configured error."""

    self.wait_calls.append(timeout)

    if self.wait_error is not None:
      raise self.wait_error

  def kill(self) -> None:
    """Records kill calls."""

    self.killed = True


class FakeReader:
  """Small Thread test double for Mosquitto stdout reader cleanup."""

  def __init__(self, alive: bool = False) -> None:
    """Initializes reader state."""

    self.alive = alive
    self.join_calls = list()

  def join(self, timeout: float) -> None:
    """Records join calls."""

    self.join_calls.append(timeout)

  def is_alive(self) -> bool:
    """Returns the configured thread liveness."""

    return self.alive


class TestClientServer(BlockTestBase):
  """Unit tests for the ClientServer Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Resets fake MQTT state before each test."""

    FakeMQTTClient.reset()

  def _mqtt_patch(self):
    """Patches ClientServer to use the fake MQTT module."""

    return patch.object(client_server_module, 'mqtt', FakeMQTTModule)

  @staticmethod
  def _capture_send(block: ClientServer) -> list[dict[str, Any]]:
    """Captures output data sent by ClientServer."""

    sent = list()

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    block.send = send
    return sent

  @staticmethod
  def _set_buffer(block: ClientServer) -> None:
    """Initializes the receive buffer without calling prepare."""

    block._buffer_output = {topic: Queue() for topic in block._topics}

  @staticmethod
  def _set_t0(block: ClientServer) -> None:
    """Sets a deterministic start time on a ClientServer."""

    block._instance_t0 = Value('d', TestClientServer._t0)

  def test_topics_and_command_labels_normalization(self) -> None:
    """Checks the supported topic and cmd_label forms."""

    block = ClientServer(
      topics=[('t', 'value'), 'state'],
      cmd_labels=['cmd', ('t(s)', 'drive')],
      labels_to_send=['remote_cmd', ('time', 'remote_drive')],
      init_output={'t': 0, 'value': 0, 'state': 'idle'})

    self.assertEqual(block._topics, [('t', 'value'), ('state',)])
    self.assertEqual(block._last_out_val, {
      't': None,
      'value': None,
      'state': None,
    })
    self.assertEqual(block._cmd_labels, [('cmd',), ('t(s)', 'drive')])
    self.assertEqual(block._labels_to_send, {
      ('cmd',): ('remote_cmd',),
      ('t(s)', 'drive'): ('time', 'remote_drive'),
    })

  def test_labels_to_send_length_must_match_command_labels(self) -> None:
    """Checks that each command topic gets a rename topic if any is given."""

    with self.assertRaises(ValueError):
      ClientServer(cmd_labels=['cmd', 'state'],
                   labels_to_send=['remote_cmd'])

  def test_prepare_requires_matching_links(self) -> None:
    """Checks that topics and cmd_labels match the Link layout."""

    with self.assertRaises(ValueError):
      ClientServer(topics=['sensor']).prepare()

    with self.assertRaises(ValueError):
      ClientServer(cmd_labels=['cmd']).prepare()

    source = TestBlock()
    block = ClientServer(topics=['sensor'], cmd_labels=['cmd'])
    link(source, block)

    with self.assertRaises(ValueError):
      block.prepare()

    block = ClientServer(topics=['sensor'], cmd_labels=['cmd'])
    sink = TestBlock()
    link(block, sink)

    with self.assertRaises(ValueError):
      block.prepare()

  def test_prepare_configures_client_and_on_connect_subscriptions(
      self) -> None:
    """Checks MQTT client setup, connection, and topic subscriptions."""

    source = TestBlock()
    block = ClientServer(address='mqtt.local',
                         port=1884,
                         topics=['sensor'],
                         cmd_labels=['cmd'])
    sink = TestBlock()
    link(source, block)
    link(block, sink)

    with self._mqtt_patch(), patch.object(client_server_module,
                                          'time',
                                          return_value=123.456):
      block.prepare()

    client = FakeMQTTClient.instances[-1]

    self.assertEqual(client.client_id, '123.456')
    self.assertEqual(client.reconnect_delay_calls, [10])
    self.assertEqual(client.connect_calls, [('mqtt.local', 1884, 10)])
    self.assertEqual(client.loop_start_calls, 1)
    self.assertIs(block._client, client)
    self.assertEqual(set(block._buffer_output), {('sensor',)})

    client.on_connect(None, None, None, 0)

    self.assertEqual(client.subscriptions, [(str(('sensor',)), 0)])
    self.assertEqual(client.loop_start_calls, 2)

  def test_prepare_retries_refused_connections(self) -> None:
    """Checks that refused connections are retried before succeeding."""

    block = ClientServer(topics=['sensor'])
    link(block, TestBlock())
    FakeMQTTClient.connect_errors = [ConnectionRefusedError]

    with (self._mqtt_patch(),
          patch.object(client_server_module, 'sleep') as mocked_sleep):
      block.prepare()

    client = FakeMQTTClient.instances[-1]

    self.assertEqual(client.connect_calls, [
      ('localhost', 1883, 10),
      ('localhost', 1883, 10),
    ])
    mocked_sleep.assert_called_once_with(1)

  def test_prepare_translates_connection_errors(self) -> None:
    """Checks user-facing errors for unreachable or invalid addresses."""

    cases = (
      (timeout(), TimeoutError),
      (gaierror(), ValueError),
    )

    for error, expected_exception in cases:
      with self.subTest(error=type(error).__name__):
        block = ClientServer(topics=['sensor'])
        link(block, TestBlock())
        FakeMQTTClient.reset()
        FakeMQTTClient.connect_errors = [error]

        with self._mqtt_patch():
          with self.assertRaises(expected_exception):
            block.prepare()

  def test_prepare_surfaces_missing_paho(self) -> None:
    """Checks that missing paho-mqtt fails only when preparing the Block."""

    block = ClientServer(topics=['sensor'])
    link(block, TestBlock())

    with patch.object(client_server_module,
                      'mqtt',
                      OptionalModule("paho-mqtt")):
      with self.assertRaisesRegex(RuntimeError, "Missing module: paho-mqtt"):
        block.prepare()

  def test_on_message_buffers_all_data_points(self) -> None:
    """Checks decoding of broker payloads into per-topic queues."""

    block = ClientServer(topics=[('a', 'b')])
    self._set_buffer(block)

    block._on_message(None,
                      None,
                      FakeMessage(('a', 'b'),
                                  dumps([[1, 2], [10, 20]])))

    self.assertEqual(block._buffer_output[('a', 'b')].get_nowait(), [1, 10])
    self.assertEqual(block._buffer_output[('a', 'b')].get_nowait(), [2, 20])
    self.assertTrue(block._buffer_output[('a', 'b')].empty())

  def test_on_message_ignores_unpickleable_payloads(self) -> None:
    """Checks that invalid broker payloads are logged and ignored."""

    block = ClientServer(topics=['a'])
    self._set_buffer(block)
    logs = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    block.log = log
    block._on_message(None, None, FakeMessage(('a',), b'not a pickle'))

    self.assertTrue(block._buffer_output[('a',)].empty())
    self.assertEqual(len(logs), 2)
    self.assertEqual(logs[-1][0], logging.WARNING)

  def test_loop_sends_received_values_and_init_values(self) -> None:
    """Checks broker input forwarding with init values for quiet topics."""

    block = ClientServer(topics=[('a', 'b'), 'state'],
                         init_output={'state': 'idle'})
    self._set_buffer(block)
    self._set_t0(block)
    sent = self._capture_send(block)

    block._on_message(None,
                      None,
                      FakeMessage(('a', 'b'),
                                  dumps([[1, 2], [10, 20]])))

    with patch.object(client_server_module, 'time', return_value=15.0):
      block.loop()

    with patch.object(client_server_module, 'time', return_value=16.5):
      block.loop()

    self.assertEqual(sent, [
      {'a': 1, 'b': 10, 'state': 'idle', 't(s)': 5.0},
      {'a': 2, 'b': 20, 'state': 'idle', 't(s)': 6.5},
    ])

  def test_loop_reuses_last_received_values(self) -> None:
    """Checks that missing labels are completed from previous values."""

    block = ClientServer(topics=['a', 'b'])
    self._set_buffer(block)
    self._set_t0(block)
    sent = self._capture_send(block)

    block._buffer_output[('a',)].put_nowait([1])
    block._buffer_output[('b',)].put_nowait([2])

    with patch.object(client_server_module, 'time', return_value=11.0):
      block.loop()

    block._buffer_output[('a',)].put_nowait([3])

    with patch.object(client_server_module, 'time', return_value=12.0):
      block.loop()

    self.assertEqual(sent, [
      {'a': 1, 'b': 2, 't(s)': 1.0},
      {'a': 3, 'b': 2, 't(s)': 2.0},
    ])

  def test_loop_spam_sends_initial_values_without_new_message(self) -> None:
    """Checks spam mode output before any broker message is received."""

    block = ClientServer(topics=['a'],
                         init_output={'a': 0},
                         spam=True)
    self._set_buffer(block)
    self._set_t0(block)
    sent = self._capture_send(block)

    with patch.object(client_server_module, 'time', return_value=12.5):
      block.loop()

    self.assertEqual(sent, [{'a': 0, 't(s)': 2.5}])

  def test_loop_requires_init_values_for_missing_labels(self) -> None:
    """Checks that partial broker input needs init or previous values."""

    block = ClientServer(topics=['a', 'b'])
    self._set_buffer(block)
    block._buffer_output[('a',)].put_nowait([1])

    with self.assertRaises(ValueError):
      block.loop()

  def test_loop_publishes_complete_commands(self) -> None:
    """Checks command grouping, renaming, and payload serialization."""

    block = ClientServer(
      cmd_labels=[('t(s)', 'cmd'), 'state'],
      labels_to_send=[('time', 'remote_cmd'), 'remote_state'])
    client = FakeMQTTClient(client_id='publisher')
    block._client = client

    def recv_all_data_raw() -> list[dict[str, list[Any]]]:
      return [
        {'t(s)': [0, 1]},
        {
          't(s)': [0, 1],
          'cmd': [10, 20],
          'state': ['idle', 'run'],
        },
      ]

    block.recv_all_data_raw = recv_all_data_raw

    block.loop()

    self.assertEqual([topic for topic, _, _ in client.publications], [
      str(('time', 'remote_cmd')),
      str(('remote_state',)),
    ])
    self.assertEqual([qos for _, _, qos in client.publications], [0, 0])
    self.assertEqual(loads(client.publications[0][1]), [[0, 1], [10, 20]])
    self.assertEqual(loads(client.publications[1][1]), [['idle', 'run']])

  def test_loop_does_not_publish_incomplete_commands(self) -> None:
    """Checks that a topic is published only from a complete input dict."""

    block = ClientServer(cmd_labels=[('t(s)', 'cmd')])
    client = FakeMQTTClient(client_id='publisher')
    block._client = client
    block.recv_all_data_raw = lambda: [{'t(s)': [0, 1]}]

    block.loop()

    self.assertEqual(client.publications, [])

  def test_finish_stops_client_and_broker(self) -> None:
    """Checks MQTT disconnection and managed broker termination."""

    block = ClientServer(broker=True)
    client = FakeMQTTClient(client_id='client')
    proc = FakeProcess()
    reader = FakeReader()

    block._client = client
    block._proc = proc
    block._reader = reader

    block.finish()

    self.assertEqual(client.loop_stop_calls, 1)
    self.assertEqual(client.disconnect_calls, 1)
    self.assertTrue(proc.terminated)
    self.assertEqual(proc.wait_calls, [15])
    self.assertFalse(proc.killed)
    self.assertTrue(block._stop_mosquitto)
    self.assertEqual(reader.join_calls, [0.2])

  def test_finish_kills_broker_after_timeout(self) -> None:
    """Checks that an unresponsive managed broker is killed."""

    proc = FakeProcess(wait_error=TimeoutExpired(cmd='mosquitto',
                                                 timeout=15))
    block = ClientServer(broker=True)
    block._proc = proc

    block.finish()

    self.assertTrue(proc.terminated)
    self.assertEqual(proc.wait_calls, [15])
    self.assertTrue(proc.killed)

  def test_launch_mosquitto_starts_expected_subprocess(self) -> None:
    """Checks the Mosquitto command line without starting a real broker."""

    block = ClientServer(broker=True, port=1885)
    proc = FakeProcess()

    with patch.object(client_server_module, 'Popen', return_value=proc) as pop:
      block._launch_mosquitto()

    pop.assert_called_once_with(['mosquitto', '-p', '1885'],
                                stdout=client_server_module.PIPE,
                                stderr=client_server_module.STDOUT)
    self.assertIs(block._proc, proc)

  def test_launch_mosquitto_reports_missing_executable(self) -> None:
    """Checks the error raised when Mosquitto is not installed."""

    block = ClientServer(broker=True)

    with patch.object(client_server_module,
                      'Popen',
                      side_effect=FileNotFoundError):
      with self.assertRaisesRegex(FileNotFoundError,
                                  "Mosquitto is not installed"):
        block._launch_mosquitto()

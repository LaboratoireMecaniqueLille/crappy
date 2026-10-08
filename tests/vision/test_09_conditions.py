# coding: utf-8

from multiprocessing import Event
from threading import Event as ThreadEvent, Thread
from time import monotonic
from unittest.mock import patch

import numpy as np

from crappy.links import ImageLink

from .vision_test_base import StubVisionBlock, VisionTestBase


class TestImageConditions(VisionTestBase):
  """Checks timed, any-input notification without changing image ownership."""

  def setUp(self) -> None:
    super().setUp()
    self._receivers = list()

  def tearDown(self) -> None:
    # Stop test threads before the shared-memory buffers are released.
    try:
      for block, thread in self._receivers:
        block._stop_event.set()
        with block.img_condition:
          block.img_condition.notify_all()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
    finally:
      super().tearDown()

  def prepare_graph(self, source_count=1, consumer_count=1):
    """Prepares real shared buffers for a small fan-in/fan-out graph."""

    sources = [StubVisionBlock(img_shape=(2, 3), img_dtype='uint8')
               for _ in range(source_count)]
    consumers = [StubVisionBlock() for _ in range(consumer_count)]
    links = [[ImageLink(source, consumer,
                        name=f'images-{source.name}-{consumer.name}')
              for consumer in consumers]
             for source in sources]
    self.make_manager(*sources)
    for consumer in consumers:
      self.set_prepare_sync(consumer)
    for source in sources:
      source.set_shared_objects()
      source.prepare()
    for consumer in consumers:
      consumer.prepare()
    return sources, consumers, links

  def start_receive(self, block, timeout=5.0):
    """Starts a receive and signals each actual Condition wait entry."""

    entered = ThreadEvent()
    results, errors = list(), list()
    real_wait = block.img_condition.wait

    def wait(timeout=None):
      self.assertIsNotNone(timeout)
      self.assertGreater(timeout, 0)
      entered.set()
      return real_wait(timeout)

    patcher = patch.object(block.img_condition, 'wait', side_effect=wait)
    patcher.start()
    self.addCleanup(patcher.stop)

    def receive():
      try:
        results.append(block.receive_imgs(timeout=timeout))
      except BaseException as exc:
        errors.append(exc)

    thread = Thread(target=receive, daemon=True)
    self._receivers.append((block, thread))
    thread.start()
    self.assertTrue(entered.wait(timeout=1.0))
    return thread, entered, results, errors

  @staticmethod
  def publish(source, image_id=1):
    source.send_img({'ImageUniqueID': image_id, 't(s)': 0.1},
                    np.full((2, 3), image_id, dtype=np.uint8))

  def test_wait_wakes_on_new_image(self) -> None:
    sources, consumers, links = self.prepare_graph()
    consumer = consumers[0]
    thread, _, results, errors = self.start_receive(consumer)

    self.publish(sources[0])
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [[links[0][0].name]])
    np.testing.assert_array_equal(consumer.last_received[links[0][0].name].img,
                                  np.ones((2, 3), dtype=np.uint8))

  def test_any_input_can_wake_receiver(self) -> None:
    sources, consumers, links = self.prepare_graph(source_count=2)
    consumer = consumers[0]
    self.assertIs(sources[0]._out_img_conditions[0], consumer.img_condition)
    self.assertIs(sources[1]._out_img_conditions[0], consumer.img_condition)
    thread, _, results, errors = self.start_receive(consumer)

    # The first input stays idle: there must be no per-input sequential wait.
    self.publish(sources[1])
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [[links[1][0].name]])
    self.assertEqual(consumer.last_received[links[0][0].name].id, -1)

  def test_publication_wakes_every_consumer(self) -> None:
    sources, consumers, links = self.prepare_graph(consumer_count=2)
    self.assertIsNot(consumers[0].img_condition, consumers[1].img_condition)
    receivers = [self.start_receive(consumer) for consumer in consumers]

    self.publish(sources[0])

    for j, (thread, _, results, errors) in enumerate(receivers):
      thread.join(timeout=1.0)
      self.assertFalse(thread.is_alive())
      self.assertEqual(errors, [])
      self.assertEqual(results, [[links[0][j].name]])

  def test_frame_published_before_wait_is_not_lost(self) -> None:
    sources, consumers, links = self.prepare_graph()
    self.publish(sources[0])

    with patch.object(consumers[0].img_condition, 'wait') as wait:
      self.assertEqual(consumers[0].receive_imgs(timeout=0.1),
                       [links[0][0].name])
    wait.assert_not_called()

  def test_spurious_notification_does_not_return_a_frame(self) -> None:
    sources, consumers, links = self.prepare_graph()
    consumer = consumers[0]
    thread, entered, results, errors = self.start_receive(consumer)
    entered.clear()

    with consumer.img_condition:
      consumer.img_condition.notify_all()
    self.assertTrue(entered.wait(timeout=1.0))
    self.assertTrue(thread.is_alive())
    self.assertEqual(results, [])

    self.publish(sources[0])
    thread.join(timeout=1.0)
    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [[links[0][0].name]])

  def test_wait_times_out_without_image(self) -> None:
    _, consumers, _ = self.prepare_graph()
    started = monotonic()

    self.assertEqual(consumers[0].receive_imgs(timeout=0.03), [])

    self.assertGreaterEqual(monotonic() - started, 0.02)
    self.assertLess(monotonic() - started, 1.0)

  def test_stop_and_error_flags_wake_without_copying(self) -> None:
    for flag in ('stop', 'error', 'barrier'):
      with self.subTest(flag=flag):
        _, consumers, _ = self.prepare_graph()
        consumer = consumers[0]
        consumer._raise_event = Event()
        thread, _, results, errors = self.start_receive(consumer)

        if flag == 'stop':
          consumer._stop_event.set()
        elif flag == 'error':
          consumer._raise_event.set()
        else:
          consumer._ready_barrier.abort()
        with consumer.img_condition:
          consumer.img_condition.notify_all()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [[]])

  def test_timeout_notices_stop_without_notification(self) -> None:
    _, consumers, _ = self.prepare_graph()
    consumer = consumers[0]
    thread, _, results, errors = self.start_receive(consumer, timeout=0.05)

    consumer._stop_event.set()
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [[]])

  def test_stop_takes_priority_over_pending_frame(self) -> None:
    sources, consumers, links = self.prepare_graph()
    self.publish(sources[0])
    consumers[0]._stop_event.set()

    self.assertEqual(consumers[0].receive_imgs(timeout=0.1), [])
    self.assertEqual(consumers[0].last_received[links[0][0].name].id, -1)

  def test_wait_validates_timeout_and_input_state(self) -> None:
    _, consumers, _ = self.prepare_graph()
    consumer = consumers[0]
    for timeout in (-1, float('inf'), float('nan')):
      with self.subTest(timeout=timeout), self.assertRaises(ValueError):
        consumer.receive_imgs(timeout=timeout)
    for timeout in (None, 'invalid'):
      with self.subTest(timeout=timeout), self.assertRaises(TypeError):
        consumer.receive_imgs(timeout=timeout)

    consumer._in_link_data[0].img_id = None
    with self.assertRaisesRegex(ValueError, 'image ID'):
      consumer.receive_imgs(timeout=0.1)
    consumer._in_link_data.clear()
    with self.assertRaisesRegex(ValueError, 'all input ImageLinks'):
      consumer.receive_imgs(timeout=0.1)

  def test_condition_failure_propagates(self) -> None:
    _, consumers, _ = self.prepare_graph()
    with (patch.object(consumers[0].img_condition, 'wait_for',
                       side_effect=OSError('condition failure')),
          self.assertRaisesRegex(OSError, 'condition failure')):
      consumers[0].receive_imgs(timeout=0.1)

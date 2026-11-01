"""Public validation and protected state for this feature."""

from unittest.mock import patch
from crappy.links import ImageLink, Link, link_graph
from crappy.blocks.vision import ImageRecorder
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock


class TestLinkValidation(BlockTestBase):
  def test_link_arguments_fail_before_opening_a_pipe(self):
    upstream, downstream = TestBlock(), TestBlock()
    with patch('crappy.links.link.Pipe') as pipe:
      cases = ({'modifiers': [object()]},
               {'allow_parallel': 'yes'}, {'name': ''})
      for options in cases:
        with self.subTest(options=options):
          with self.assertRaises((TypeError, ValueError)):
            Link(upstream, downstream, **options)
      pipe.assert_not_called()

  def test_image_link_rejects_nonvision_blocks_before_graph_registration(self):
    upstream, downstream = TestBlock(), TestBlock()
    vision = ImageRecorder(save_backend='npy')
    with patch.object(link_graph, 'add_edge') as add_edge:
      for source, target in ((upstream, vision), (vision, downstream)):
        with self.subTest(source=source, target=target):
          with self.assertRaises(NotImplementedError):
            ImageLink(source, target)
      add_edge.assert_not_called()

  def test_rejected_data_does_not_interrupt_link_delivery(self):
    upstream, downstream = TestBlock(), TestBlock()
    connection = Link(upstream, downstream)
    upstream.send({'x': 1})
    self.assertEqual(connection.recv(), {'x': 1})
    with self.assertRaises((TypeError, ValueError)):
      upstream.send({1: 'invalid label'})
    self.assertFalse(connection.poll())
    upstream.send({'x': 2})
    self.assertEqual(connection.recv(), {'x': 2})

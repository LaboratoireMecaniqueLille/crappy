# coding: utf-8

from unittest import TestCase

from crappy.links import GraphStructureError, LinkGraph


class FirstBlock:
  """First dummy Block type stored in graph nodes."""


class SecondBlock:
  """Second dummy Block type stored in graph nodes."""


class TestLinkGraph(TestCase):
  """Tests LinkGraph mutations and the invariants they preserve."""

  def setUp(self) -> None:
    """Creates an empty graph for each test."""

    self.graph = LinkGraph()

  def add_nodes(self, *names: str) -> None:
    """Adds dummy nodes with the requested names."""

    for name in names:
      self.graph.add_node(name, FirstBlock)

  def test_add_node(self) -> None:
    """Tests node storage and independent adjacency lists."""

    self.graph.add_node('first', FirstBlock)
    self.graph.add_node('second', SecondBlock)

    self.assertEqual(tuple(self.graph._nodes), ('first', 'second'))
    self.assertEqual(self.graph._nodes['first'].name, 'first')
    self.assertIs(self.graph._nodes['first'].block_type, FirstBlock)
    self.assertIs(self.graph._nodes['second'].block_type, SecondBlock)
    self.assertIsNot(self.graph._out_edges['first'],
                     self.graph._out_edges['second'])
    self.assertIsNot(self.graph._in_edges['first'],
                     self.graph._in_edges['second'])

  def test_add_node_validates_arguments(self) -> None:
    """Tests node name and type validation without partial registration."""

    for name in (None, 1, tuple()):
      with self.subTest(name=name):
        with self.assertRaises(TypeError):
          self.graph.add_node(name, FirstBlock)

    with self.assertRaises(ValueError):
      self.graph.add_node('', FirstBlock)
    with self.assertRaises(TypeError):
      self.graph.add_node('first', FirstBlock())

    self.assertEqual(self.graph._nodes, dict())
    self.assertEqual(self.graph._out_edges, dict())
    self.assertEqual(self.graph._in_edges, dict())

  def test_add_node_rejects_duplicate_names(self) -> None:
    """Tests that duplicate registration preserves the original node."""

    self.graph.add_node('first', FirstBlock)

    with self.assertRaises(GraphStructureError):
      self.graph.add_node('first', SecondBlock)

    self.assertEqual(len(self.graph._nodes), 1)
    self.assertIs(self.graph._nodes['first'].block_type, FirstBlock)

  def test_add_edge(self) -> None:
    """Tests edge storage, filtering, and adjacency registration."""

    self.add_nodes('source', 'target')
    self.graph.add_edge('data', 'source', 'target', 'link')
    self.graph.add_edge('image', 'source', 'target', 'image')

    data_edge = self.graph._edges['data']
    image_edge = self.graph._edges['image']

    self.assertEqual((data_edge.name, data_edge.source, data_edge.target,
                      data_edge.kind),
                     ('data', 'source', 'target', 'link'))
    self.assertIs(self.graph._out_edges['source'][0], data_edge)
    self.assertIs(self.graph._in_edges['target'][0], data_edge)
    self.assertIs(self.graph._out_edges['source'][1], image_edge)
    self.assertIs(self.graph._in_edges['target'][1], image_edge)
    self.assertEqual(self.graph.link_names(), ('data', 'image'))
    self.assertEqual(self.graph.link_names('link'), ('data',))
    self.assertEqual(self.graph.link_names('image'), ('image',))

  def test_add_edge_validates_arguments(self) -> None:
    """Tests edge argument validation without partial registration."""

    self.add_nodes('source', 'target')

    invalid_calls = (
        (TypeError, (None, 'source', 'target', 'link'), dict()),
        (ValueError, ('', 'source', 'target', 'link'), dict()),
        (TypeError, ('edge', None, 'target', 'link'), dict()),
        (ValueError, ('edge', '', 'target', 'link'), dict()),
        (TypeError, ('edge', 'source', None, 'link'), dict()),
        (ValueError, ('edge', 'source', '', 'link'), dict()),
        (ValueError, ('edge', 'source', 'target', 'other'), dict()),
        (TypeError, ('edge', 'source', 'target', 'link'),
         {'allow_parallel': 1}),
    )

    for exception, args, kwargs in invalid_calls:
      with self.subTest(args=args, kwargs=kwargs):
        with self.assertRaises(exception):
          self.graph.add_edge(*args, **kwargs)

    self.assertEqual(self.graph._edges, dict())
    self.assertEqual(self.graph._out_edges['source'], list())
    self.assertEqual(self.graph._in_edges['target'], list())

  def test_add_edge_requires_registered_endpoints(self) -> None:
    """Tests that both edge endpoints must already exist."""

    self.graph.add_node('source', FirstBlock)

    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('missing-target', 'source', 'target', 'link')
    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('missing-source', 'other', 'source', 'link')

    self.assertEqual(self.graph._edges, dict())

  def test_add_edge_rejects_duplicate_names(self) -> None:
    """Tests global edge-name uniqueness without partial registration."""

    self.add_nodes('first', 'second', 'third')
    self.graph.add_edge('edge', 'first', 'second', 'link')

    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('edge', 'first', 'third', 'link')

    self.assertEqual(self.graph.link_names(), ('edge',))
    self.assertEqual(self.graph.successors('first'), ('second',))
    self.assertEqual(self.graph.predecessors('third'), tuple())

  def test_parallel_regular_edges_require_opt_in(self) -> None:
    """Tests the explicit override for same-kind, same-endpoint edges."""

    self.add_nodes('source', 'target')
    self.graph.add_edge('first', 'source', 'target', 'link')

    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('rejected', 'source', 'target', 'link')

    self.graph.add_edge('second', 'source', 'target', 'link',
                        allow_parallel=True)

    self.assertEqual(self.graph.link_names(), ('first', 'second'))
    self.assertEqual(len(self.graph._out_edges['source']), 2)
    self.assertEqual(len(self.graph._in_edges['target']), 2)

  def test_parallel_image_edges_can_be_enabled_on_standalone_graphs(self) \
      -> None:
    """Tests the generic graph override that ImageLink does not expose."""

    self.add_nodes('source', 'target')
    self.graph.add_edge('first', 'source', 'target', 'image')
    self.graph.add_edge('second', 'source', 'target', 'image',
                        allow_parallel=True)

    self.assertEqual(self.graph.link_names('image'), ('first', 'second'))

  def test_regular_edges_may_form_cycles(self) -> None:
    """Tests that regular feedback loops and self-loops remain valid."""

    self.add_nodes('first', 'second')
    self.graph.add_edge('forward', 'first', 'second', 'link')
    self.graph.add_edge('backward', 'second', 'first', 'link')
    self.graph.add_edge('self', 'first', 'first', 'link')

    self.assertEqual(self.graph.link_names('link'),
                     ('forward', 'backward', 'self'))

  def test_image_edges_must_remain_acyclic(self) -> None:
    """Tests rejection of direct and indirect image-only cycles."""

    self.add_nodes('first', 'second', 'third')
    self.graph.add_edge('first-image', 'first', 'second', 'image')
    self.graph.add_edge('second-image', 'second', 'third', 'image')

    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('cycle', 'third', 'first', 'image')
    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('self', 'first', 'first', 'image')
    with self.assertRaises(GraphStructureError):
      self.graph.add_edge('parallel', 'first', 'second', 'image')

    self.assertEqual(self.graph.link_names('image'),
                     ('first-image', 'second-image'))

  def test_regular_paths_do_not_create_image_cycles(self) -> None:
    """Tests that cycle validation considers only ImageLinks."""

    self.add_nodes('first', 'second', 'third')
    self.graph.add_edge('data', 'first', 'second', 'link')
    self.graph.add_edge('image', 'second', 'first', 'image')
    self.graph.add_edge('other-image', 'first', 'third', 'image')

    self.assertEqual(self.graph.link_names(),
                     ('data', 'image', 'other-image'))

  def test_rename_node(self) -> None:
    """Tests renaming a node with incoming, outgoing, and self-loop edges."""

    self.add_nodes('first', 'middle', 'last')
    self.graph.add_edge('incoming', 'first', 'middle', 'link')
    self.graph.add_edge('outgoing', 'middle', 'last', 'image')
    self.graph.add_edge('self', 'middle', 'middle', 'link')

    self.graph.rename_node('middle', 'renamed')

    self.assertNotIn('middle', self.graph._nodes)
    self.assertNotIn('middle', self.graph._out_edges)
    self.assertNotIn('middle', self.graph._in_edges)
    self.assertIs(self.graph._nodes['renamed'].block_type, FirstBlock)
    self.assertEqual((self.graph._edges['incoming'].source,
                      self.graph._edges['incoming'].target),
                     ('first', 'renamed'))
    self.assertEqual((self.graph._edges['outgoing'].source,
                      self.graph._edges['outgoing'].target),
                     ('renamed', 'last'))
    self.assertEqual((self.graph._edges['self'].source,
                      self.graph._edges['self'].target),
                     ('renamed', 'renamed'))
    self.assertEqual(self.graph.predecessors('renamed'),
                     ('first', 'renamed'))
    self.assertEqual(self.graph.successors('renamed'),
                     ('last', 'renamed'))

  def test_rename_node_validates_names(self) -> None:
    """Tests rename validation leaves the graph unchanged."""

    self.add_nodes('first', 'second')
    first_node = self.graph._nodes['first']

    invalid_calls = (
        (TypeError, None, 'renamed'),
        (ValueError, '', 'renamed'),
        (KeyError, 'missing', 'renamed'),
        (TypeError, 'first', None),
        (ValueError, 'first', ''),
        (GraphStructureError, 'first', 'second'),
    )

    for exception, old_name, new_name in invalid_calls:
      with self.subTest(old_name=old_name, new_name=new_name):
        with self.assertRaises(exception):
          self.graph.rename_node(old_name, new_name)

    self.assertEqual(tuple(self.graph._nodes), ('first', 'second'))
    self.assertIs(self.graph._nodes['first'], first_node)

  def test_rename_to_same_name_is_a_noop(self) -> None:
    """Tests that assigning the current name preserves the node object."""

    self.graph.add_node('first', FirstBlock)
    node = self.graph._nodes['first']

    self.graph.rename_node('first', 'first')

    self.assertIs(self.graph._nodes['first'], node)

  def test_reset(self) -> None:
    """Tests idempotent reset and reuse of former node and edge names."""

    self.add_nodes('source', 'target')
    self.graph.add_edge('edge', 'source', 'target', 'link')

    self.graph.reset()
    self.graph.reset()

    self.assertEqual(self.graph._nodes, dict())
    self.assertEqual(self.graph._edges, dict())
    self.assertEqual(self.graph._out_edges, dict())
    self.assertEqual(self.graph._in_edges, dict())

    self.add_nodes('source', 'target')
    self.graph.add_edge('edge', 'source', 'target', 'link')
    self.assertEqual(self.graph.link_names(), ('edge',))

  def test_link_name_filter_validation(self) -> None:
    """Tests rejection of an unsupported Link-kind filter."""

    with self.assertRaises(ValueError):
      self.graph.link_names('other')

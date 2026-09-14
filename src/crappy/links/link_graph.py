# coding: utf-8

from collections import deque
from dataclasses import dataclass
from typing import Literal
from platform import system
from pathlib import Path

from .._global import OptionalModule

try:
  from graphviz import Digraph
except (ModuleNotFoundError, ImportError):
  Digraph = OptionalModule("graphviz")


@dataclass
class Node:
  """Description of a Block registered in a :class:`LinkGraph`.

  Args:
    name: Unique name of the Block.
    block_type: Concrete type of the Block. The type, rather than the instance,
      is stored so that the graph does not keep Blocks alive.
  """

  name: str
  block_type: type


@dataclass
class Edge:
  """Description of a directed connection registered in a
  :class:`LinkGraph`.

  Args:
    name: Name shared with the corresponding Link object.
    source: Name of the upstream Block.
    target: Name of the downstream Block.
    kind: ``'link'`` for dictionary data or ``'image'`` for image data.

  The source and target fields are updated when either endpoint is renamed.
  """

  name: str
  source: str
  target: str
  kind: Literal["link", "image"]


class GraphStructureError(ValueError):
  """Raised when the Block connection graph has an invalid structure."""


class LinkGraph:
  """Directed graph describing the connections between instantiated Blocks.

  Node names and edge names are unique within their respective namespaces.
  Multiple edges of the same kind may connect the same ordered pair of Blocks
  only when explicitly enabled while adding the extra edges. Regular Links may
  form cycles, including self-loops. ImageLinks may not make the image-only
  subgraph cyclic.

  Crappy maintains one module-level ``link_graph`` instance in the main
  process. Individual :class:`LinkGraph` instances can also be useful for
  validating graph-building code independently.

  .. versionadded:: 2.1.0
  """

  def __init__(self) -> None:
    """Initializes an empty graph."""

    self._nodes: dict[str, Node] = dict()
    self._edges: dict[str, Edge] = dict()
    self._out_edges: dict[str, list[Edge]] = dict()
    self._in_edges: dict[str, list[Edge]] = dict()

  @property
  def nodes(self) -> dict[str, Node]:
    """Mapping of registered Block names to their graph nodes.

    The mapping preserves Block insertion order. It is exposed for inspection,
    graph changes should be made through the dedicated methods.
    """

    return self._nodes

  def add_node(self, name: str, block_type: type) -> None:
    """Adds a Block description to the graph.

    Args:
      name: Unique, non-empty name of the Block.
      block_type: Concrete type of the Block.

    Raises:
      TypeError: If *name* is not a string or *block_type* is not a type.
      ValueError: If *name* is empty.
      GraphStructureError: If *name* is already registered.
    """

    # Basic sanity checks on the arguments
    if not isinstance(name, str):
      raise TypeError("The Block's name must be a string")
    if not name:
      raise ValueError("The Block's name must be a non-empty string")
    if not isinstance(block_type, type):
      raise TypeError("The Block's type must be a type")

    # Adding the node except if it already exists
    if name not in self._nodes:
      self._nodes[name] = Node(name, block_type)
      self._out_edges[name] = list()
      self._in_edges[name] = list()
    else:
      raise GraphStructureError(f"Cannot add Block {name} as a block with the "
                                f"same name is already present in the graph")

  def add_edge(self,
               name: str,
               source: str,
               target: str,
               kind: Literal["link", "image"],
               allow_parallel: bool = False) -> None:
    """Adds a directed Link description to the graph.

    Args:
      name: Unique, non-empty name of the Link or ImageLink.
      source: Name of the already registered upstream Block.
      target: Name of the already registered downstream Block.
      kind: ``'link'`` for a regular Link or ``'image'`` for an ImageLink.
      allow_parallel: Allows another edge of the same kind with the same source
        and target. This option does not relax Link-name uniqueness. The
        public ImageLink API does not enable parallel edges.

    Raises:
      TypeError: If a name is not a string or *allow_parallel* is not a
        boolean.
      ValueError: If a name is empty or *kind* is invalid.
      GraphStructureError: If an endpoint is missing, the Link name is already
        registered, a parallel connection was not allowed, or an ImageLink
        would create a cycle.
    """

    # Basic sanity checks on the arguments
    if not isinstance(name, str):
      raise TypeError("The Link's name must be a string")
    if not name:
      raise ValueError("The Link's name must be a non-empty string")
    if not isinstance(source, str):
      raise TypeError("The name of the source Block must be a string")
    if not source:
      raise ValueError("The name of the source Block must be a non-empty "
                       "string")
    if not isinstance(target, str):
      raise TypeError("The name of the target Block must be a string")
    if not target:
      raise ValueError("The name of the target Block must be a non-empty "
                       "string")
    if kind not in ("link", "image"):
      raise ValueError("The Link's kind must be either 'link' or 'image'")
    if not isinstance(allow_parallel, bool):
      raise TypeError("allow_parallel must be a boolean")

    # The source and target Blocks must already be registered
    if source not in self._nodes:
      raise GraphStructureError("Cannot add Link from a Block that is not yet "
                                "registered in the graph")
    if target not in self._nodes:
      raise GraphStructureError("Cannot add Link to a Block that is not yet "
                                "registered in the graph")

    # The new Link shouldn't be already registered
    if name in self._edges:
      raise GraphStructureError(f"A Link with the same name ({name}) is "
                                f"already present in the graph")

    # A Link with a given source, target and type should be unique, except
    # when parallel Links are explicitly allowed
    if (not allow_parallel and
        any(edge.source == source and
            edge.target == target and
            edge.kind == kind for edge in self._edges.values())):
      raise GraphStructureError(f"A Link with the same source ({source}), "
                                f"same target ({target}) and same type "
                                f"({kind}) is"
                                f" already registered")

    # Check that the new Link doesn't make the image Link graph cyclic
    if kind == 'image' and (source == target or
                            source in self.descendants(target, kind='image')):
      raise GraphStructureError(f"Adding an image Link from {source} to "
                                f"{target} would make the image Link "
                                f"connection graph cyclic, operation "
                                f"forbidden!")

    # Registering the new Link
    edge = Edge(name, source, target, kind)
    self._edges[name] = edge
    self._out_edges[source].append(edge)
    self._in_edges[target].append(edge)

  def descendants(self,
                  source: str,
                  kind: Literal["link",
                                "image"] | None = None) -> tuple[str, ...]:
    """Returns all Blocks reachable from a source Block.

    Args:
      source: Name of the Block from which to start.
      kind: Optional edge-kind filter.

    Returns:
      Reachable Block names in breadth-first discovery order, without
      duplicates.

    Raises:
      KeyError: If *source* is not registered.
    """

    # Set containing the visited Blocks
    visited = {source}
    # Buffer for iterating over the Links
    queue = deque([source])
    # List containing the descendants
    ret = list()

    # Iterating until we've exhausted all the Links from the source iteratively
    while queue:
      node = queue.popleft()

      # Adding the new downstream Blocks if they haven't already been visited
      for edge in self._out_edges[node]:
        if kind is not None and edge.kind != kind:
          continue

        # Adding the discovered Block to all the relevant buffers
        ret.append(edge.target)
        if edge.target not in visited:
          visited.add(edge.target)
          queue.append(edge.target)

    # Use dict to remove duplicate values
    return tuple(dict.fromkeys(ret))

  def successors(self,
                 source: str,
                 kind: Literal["link",
                               "image"] | None = None) -> tuple[str, ...]:
    """Returns the Blocks directly downstream of a source Block.

    Args:
      source: Name of the source Block.
      kind: Optional edge-kind filter.

    Returns:
      Directly downstream Block names in Link insertion order, without
      duplicates.

    Raises:
      KeyError: If *source* is not registered.
    """

    return tuple(dict.fromkeys(edge.target for edge in self._out_edges[source]
                               if kind is None or edge.kind == kind))

  def ancestors(self,
                target: str,
                kind: Literal["link",
                              "image"] | None = None) -> tuple[str, ...]:
    """Returns all Blocks from which a target Block is reachable.

    Args:
      target: Name of the Block from which to search backwards.
      kind: Optional edge-kind filter.

    Returns:
      Reachable upstream Block names in breadth-first discovery order, without
      duplicates.

    Raises:
      KeyError: If *target* is not registered.
    """

    # Set containing the visited Blocks
    visited = {target}
    # Buffer for iterating over the Links
    queue = deque([target])
    # List containing the ancestors
    ret = list()

    # Iterating until we've exhausted all the Links to the target iteratively
    while queue:
      node = queue.popleft()

      # Adding the new upstream Blocks if they haven't already been visited
      for edge in self._in_edges[node]:
        if kind is not None and edge.kind != kind:
          continue

        # Adding the discovered Block to all the relevant buffers
        ret.append(edge.source)
        if edge.source not in visited:
          visited.add(edge.source)
          queue.append(edge.source)

    # Use dict to remove duplicate values
    return tuple(dict.fromkeys(ret))

  def predecessors(self,
                   target: str,
                   kind: Literal["link",
                                 "image"] | None = None) -> tuple[str, ...]:
    """Returns the Blocks directly upstream of a target Block.

    Args:
      target: Name of the target Block.
      kind: Optional edge-kind filter.

    Returns:
      Directly upstream Block names in Link insertion order, without
      duplicates.

    Raises:
      KeyError: If *target* is not registered.
    """

    return tuple(dict.fromkeys(edge.source for edge in self._in_edges[target]
                               if kind is None or edge.kind == kind))

  def img_sources(self) -> tuple[str, ...]:
    """Returns image-producing roots in the image-only subgraph.

    A Block is an image source when it has at least one outgoing ImageLink and
    no incoming ImageLink.

    Returns:
      Image-source names in Block insertion order.
    """

    return tuple(name for name in self._nodes
                 if any(edge.kind == "image" for edge in self._out_edges[name])
                 and not any(edge.kind == "image"
                             for edge in self._in_edges[name]))

  def rename_node(self, node: str, new_name: str) -> None:
    """Renames a Block and updates every incident edge atomically.

    Args:
      node: Current name of the Block.
      new_name: New unique, non-empty name for the Block.

    Raises:
      TypeError: If either name is not a string.
      ValueError: If either name is empty.
      KeyError: If *node* is not registered.
      GraphStructureError: If *new_name* belongs to another Block.
    """

    if not isinstance(node, str):
      raise TypeError("The current Block name must be a string")
    if not node:
      raise ValueError("The current Block name must be a non-empty string")

    # Check that the node to rename actually exists
    if node not in self._nodes:
      raise KeyError("The Block to rename isn't registered, aborting")

    # Checking the validity of the new name
    if not isinstance(new_name, str):
      raise TypeError("The new Block name must be a string")
    if not new_name:
      raise ValueError("The new Block name must be a non-empty string")

    # Nothing to do if the new name is identical
    if node == new_name:
      return

    # Cannot assign a new name that is already in use
    if new_name in self._nodes:
      raise GraphStructureError(f"The name {new_name} is already registered, "
                                f"aborting")

    # Replace the old node with a new one
    old_node = self._nodes.pop(node)
    self._nodes[new_name] = Node(new_name, old_node.block_type)

    # Move the existing edges to a new location
    self._out_edges[new_name] = self._out_edges.pop(node)
    self._in_edges[new_name] = self._in_edges.pop(node)

    # Update the Block's name wherever needed
    for edge in self._out_edges[new_name]:
      edge.source = new_name
    for edge in self._in_edges[new_name]:
      edge.target = new_name

  def link_names(self,
                 kind: Literal["link",
                               "image"] | None = None) -> tuple[str, ...]:
    """Returns registered Link names in insertion order.

    Args:
      kind: If given, only returns regular Link names (``'link'``) or
        ImageLink names (``'image'``).

    Raises:
      ValueError: If *kind* is not ``'link'``, ``'image'``, or :obj:`None`.
    """

    if kind is None:
      return tuple(self._edges.keys())
    elif kind == "image":
      return tuple(edge.name for edge in self._edges.values()
                   if edge.kind == "image")
    elif kind == "link":
      return tuple(edge.name for edge in self._edges.values()
                   if edge.kind == "link")
    else:
      raise ValueError("kind must be either 'image', 'link', or None")

  def reset(self) -> None:
    """Removes every node and edge, returning the graph to its initial state.

    This operation is idempotent. The module-level graph is reset together
    with :meth:`crappy.blocks.Block.reset`.
    """

    self._nodes.clear()
    self._edges.clear()
    self._out_edges.clear()
    self._in_edges.clear()

  def display(self,
              links: bool = True,
              image_links: bool = True) -> None:
    """Displays the connectivity graph between Blocks.

    Args:
      links: If True, displays the regular Links.
      image_links: If True, displays the ImageLinks.
    """

    if not isinstance(links, bool):
      raise TypeError("links must be a boolean")
    if not isinstance(image_links, bool):
      raise TypeError("image_links must be a boolean")
    if not links and not image_links:
      raise ValueError("At least one Link type must be displayed")

    # Getting the path to Crappy's temporary folder
    if system() in ('Linux', 'Darwin'):
      graph_path = Path('/tmp/crappy')
    elif system() == 'Windows':
      graph_path = Path.home() / 'AppData' / 'Local' / 'Temp' / 'crappy'
    else:
      graph_path = None

    graph = Digraph(name="Crappy",
                    filename="crappy_graph",
                    format='pdf',
                    directory=graph_path)
    graph.attr(rankdir="LR")

    # Add all Blocks, including isolated ones
    for node in self._nodes.values():
      graph.node(node.name,
                 label=f"{node.name}\n({node.block_type.__name__})")

    # Add the selected Links
    for edge in self._edges.values():

      if edge.kind == "link" and links:
        graph.edge(edge.source,
                   edge.target,
                   label=edge.name,
                   style="solid")

      elif edge.kind == "image" and image_links:
        graph.edge(edge.source,
                   edge.target,
                   label=edge.name,
                   style="dashed",
                   penwidth="2")

    graph.view(cleanup=True)


# The graph is managed by the main Process
link_graph = LinkGraph()

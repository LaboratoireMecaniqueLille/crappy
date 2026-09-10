=====
Links
=====

Link
----
.. autoclass:: crappy.links.Link
   :members: poll, send, recv, recv_last, recv_chunk, log
   :special-members: __init__

Image Link
----------
.. autoclass:: crappy.links.ImageLink
   :special-members: __init__

Connection graph and validation
-------------------------------

Crappy builds a directed graph while Blocks, Links, and ImageLinks are
instantiated. It uses this graph to reject inconsistent connection structures
immediately, before the processes start. In particular:

* Block names must be unique among Blocks, and Link names must be unique across
  both regular Links and ImageLinks
* a second regular Link in the same direction between the same two Blocks must
  be created with ``allow_parallel=True``
* parallel ImageLinks are rejected
* the graph formed by ImageLinks must be acyclic. Regular Links may still form
  feedback loops and self-loops.

For example, two independent data channels between the same Blocks can be
declared explicitly:

.. code-block:: python

   crappy.link(source, consumer, name="measurements")
   crappy.link(source, consumer, name="status", allow_parallel=True)

Changing :attr:`crappy.blocks.Block.name` before a Block starts updates the
node and all its connections. Renaming a running Block is rejected. Calling
:meth:`crappy.blocks.Block.reset` (also available as ``crappy.reset``) clears
the graph together with the Block registry.

The current graph can be rendered as a PDF and opened with the system viewer by
calling :func:`crappy.display_graph`. Its ``links`` and ``image_links``
arguments can be used to hide either kind of connection. This function is an
alias for :meth:`crappy.links.LinkGraph.display`.

The graph is maintained automatically by Crappy. Application code normally
does not need to interact with it. The class and its exception are documented
below:

.. autoclass:: crappy.links.LinkGraph
   :members: add_node, add_edge, descendants, successors, ancestors,
             predecessors, img_sources, rename_node, link_names, reset, display
   :special-members: __init__

.. autoclass:: crappy.links.GraphStructureError

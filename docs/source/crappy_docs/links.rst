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

ImageLinks connect :class:`~crappy.blocks.vision.VisionBlock` objects and
carry image arrays together with their metadata. Application code normally
creates one through :func:`crappy.img_link`, just as :func:`crappy.link`
creates a regular Link.

Unlike a regular Link, an ImageLink is not a queue. An image-producing
VisionBlock owns a shared-memory buffer that is reused by all of its outgoing
ImageLinks. Each consumer copies the newest coherent image and metadata when
it is ready. A consumer that runs more slowly than the source can therefore
skip intermediate frames without blocking the source. Commands, processing
results, and overlays remain small dictionaries and should travel through
regular Links.

.. autoclass:: crappy.links.ImageLink
   :members: set_buffers, get_buffers, log
   :special-members: __init__

The buffer methods above are called by Crappy while preparing the Blocks. Most
user scripts only need :func:`crappy.img_link`. Custom VisionBlocks exchange
images through :meth:`~crappy.blocks.vision.VisionBlock.send_img` and
:meth:`~crappy.blocks.vision.VisionBlock.receive_imgs`.

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
does not need to interact with it. The graph class, the node records returned
by its :attr:`~crappy.links.LinkGraph.nodes` property, and its exception are
documented below:

.. autoclass:: crappy.links.LinkGraph
   :members: nodes, add_node, add_edge, descendants, successors, ancestors,
             predecessors, img_sources, rename_node, link_names, reset, display
   :special-members: __init__

.. autoclass:: crappy.links.link_graph.Node
   :members:

.. autoclass:: crappy.links.GraphStructureError

=====
Links
=====

Link
----

Regular Links carry labeled dictionaries between Blocks. Their delivery and
loss behavior is explained in :doc:`../concepts/regular_links_and_image_links`.

.. autoclass:: crappy.links.Link
   :members: poll, send, recv, recv_last, recv_chunk, log
   :special-members: __init__

Image Link
----------

ImageLinks carry the newest image and its metadata between VisionBlocks. See
:doc:`../concepts/regular_links_and_image_links` for the transport comparison
and frame-skipping semantics.

.. autoclass:: crappy.links.ImageLink
   :members: set_buffers, get_buffers, log
   :special-members: __init__

The buffer methods above are called by Crappy while preparing the Blocks. Most
user scripts only need :func:`crappy.img_link`. Custom VisionBlocks exchange
images through :meth:`~crappy.blocks.vision.VisionBlock.send_img` and
:meth:`~crappy.blocks.vision.VisionBlock.receive_imgs`.

Connection graph and validation
-------------------------------

Crappy maintains the connection graph automatically. The public validation
rules, feedback behavior, and parallel-Link option are explained in
:doc:`../concepts/blocks_links_labels`. Application code normally uses
:func:`crappy.link`, :func:`crappy.img_link`, and :func:`crappy.display_graph`
rather than interacting with the graph directly. The classes below provide the
implementation reference for custom integrations and debugging.

.. autoclass:: crappy.links.LinkGraph
   :members: nodes, add_node, add_edge, descendants, successors, ancestors,
             predecessors, img_sources, rename_node, link_names, reset, display
   :special-members: __init__

.. autoclass:: crappy.links.link_graph.Node
   :members: name, block_type
   :undoc-members:

.. autoclass:: crappy.links.GraphStructureError

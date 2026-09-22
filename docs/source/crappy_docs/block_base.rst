================
Block base class
================

Subclass :class:`crappy.blocks.Block` when an experiment needs a custom task
with its own inputs, outputs, timing, setup, or cleanup.

Use :meth:`~crappy.blocks.Block.log` for messages from an individual Block.
The class-level :meth:`~crappy.blocks.Block.cls_log` method is available to
code coordinating all Blocks after the main Crappy logger has been configured.

.. autosummary::
   :nosignatures:

   crappy.blocks.Block

.. _crappy_docs/blocks:block:

Block
-----

.. automodule:: crappy.blocks.meta_block.block

.. currentmodule:: crappy.blocks.meta_block.block

.. autoclass:: Block
   :members: get_name, start_all, prepare_all, renice_all, launch_all,
             stop_all, reset, run, main, prepare, begin, loop, finish, debug,
             t0, niceness, labels, freq, display_freq, name, pausable,
             is_vision_block, add_output, add_input, log, cls_log, stop, send,
             data_available, recv_data, recv_last_data, recv_all_data,
             recv_all_data_raw
   :private-members: _cleanup
   :special-members: __init__

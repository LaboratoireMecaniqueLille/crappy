===============
Generator Paths
===============

A Generator Path defines one segment of a command sequence run by the
:class:`~crappy.blocks.Generator` Block.

.. autosummary::
   :nosignatures:

   crappy.blocks.generator_path.Conditional
   crappy.blocks.generator_path.Constant
   crappy.blocks.generator_path.Custom
   crappy.blocks.generator_path.Cyclic
   crappy.blocks.generator_path.CyclicRamp
   crappy.blocks.generator_path.Integrator
   crappy.blocks.generator_path.Ramp
   crappy.blocks.generator_path.Sine
   crappy.blocks.generator_path.meta_path.Path

.. _crappy_docs/blocks:generator paths:

Built-in Paths
--------------

Conditional
+++++++++++

.. autoclass:: crappy.blocks.generator_path.Conditional
   :members: get_cmd
   :special-members: __init__

Constant
++++++++

.. autoclass:: crappy.blocks.generator_path.Constant
   :members: get_cmd
   :special-members: __init__

Custom
++++++

.. autoclass:: crappy.blocks.generator_path.Custom
   :members: get_cmd
   :special-members: __init__

Cyclic
++++++

.. autoclass:: crappy.blocks.generator_path.Cyclic
   :members: get_cmd
   :special-members: __init__

Cyclic Ramp
+++++++++++

.. autoclass:: crappy.blocks.generator_path.CyclicRamp
   :members: get_cmd
   :special-members: __init__

Integrator
++++++++++

.. autoclass:: crappy.blocks.generator_path.Integrator
   :members: get_cmd
   :special-members: __init__

Ramp
++++

.. autoclass:: crappy.blocks.generator_path.Ramp
   :members: get_cmd
   :special-members: __init__

Sine
++++

.. autoclass:: crappy.blocks.generator_path.Sine
   :members: get_cmd
   :special-members: __init__

.. _crappy_docs/blocks:parent path:

Path base class
---------------

.. autoclass:: crappy.blocks.generator_path.meta_path.Path
   :members: get_cmd, parse_condition, log
   :special-members: __init__

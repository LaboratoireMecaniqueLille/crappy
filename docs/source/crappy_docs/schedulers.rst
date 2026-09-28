=================
Scheduler helpers
=================

A :class:`~crappy.blocks.Scheduler` runs a graph of States. Each State has
output functions and ordered transition conditions. The classes below are
conveniences: a State also accepts ordinary Python callables with the same
signatures. All classes are importable from ``crappy.blocks.schedulers``.

Start with the :doc:`Scheduler tutorial <../tutorials/scheduler>` for a
complete state graph, or browse the :ref:`Scheduler examples
<examples:scheduler state graphs>` for other patterns.

.. autosummary::
   :nosignatures:

   crappy.blocks.schedulers.State
   crappy.blocks.schedulers.Output
   crappy.blocks.schedulers.Constant
   crappy.blocks.schedulers.Ramp
   crappy.blocks.schedulers.Sine
   crappy.blocks.schedulers.Square
   crappy.blocks.schedulers.Triangle
   crappy.blocks.schedulers.FromFile
   crappy.blocks.schedulers.Condition
   crappy.blocks.schedulers.Delay
   crappy.blocks.schedulers.Compare
   crappy.blocks.schedulers.Crossing
   crappy.blocks.schedulers.AnyLabel
   crappy.blocks.schedulers.AllLabel
   crappy.blocks.schedulers.AnyCondition
   crappy.blocks.schedulers.AllCondition

State
-----

.. autoclass:: crappy.blocks.schedulers.State
   :special-members: __init__

Outputs
-------

An output receives ``(dt, data)`` and returns a dictionary of output labels
and values, or ``None``. ``dt`` is the time since entry into the current State,
``data`` maps received labels to lists of values. The Scheduler remembers the
most recent value of every output label, so a State may update only a subset.
Use the Scheduler's ``init_values`` when the first State does not set every
required label.

.. autoclass:: crappy.blocks.schedulers.Output
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Constant
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Ramp
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Sine
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Square
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Triangle
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.FromFile
   :members:
   :special-members: __init__, __call__

Conditions
----------

A condition receives the same ``(dt, data)`` arguments and returns a boolean.
Conditions are checked in their listed order, before outputs are evaluated,
the first true condition selects its target State. A condition can also be a
module-level function or callable class. Conditions and outputs with a
``reset()`` method have it called whenever their State is entered.

.. autoclass:: crappy.blocks.schedulers.Condition
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Delay
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Compare
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.Crossing
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.AnyLabel
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.AllLabel
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.AnyCondition
   :members:
   :special-members: __init__, __call__

.. autoclass:: crappy.blocks.schedulers.AllCondition
   :members:
   :special-members: __init__, __call__

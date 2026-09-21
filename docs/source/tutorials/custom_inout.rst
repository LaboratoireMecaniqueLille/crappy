.. _tutorial-custom-inout:

=====================
Create a custom InOut
=====================

This tutorial has one outcome: create an InOut that receives one command and
returns one measurement through an IOBlock.

Prerequisites
-------------

- Complete :doc:`data_acquisition`.
- Complete :doc:`command_generation` or know how to publish a command label.
- To adapt the example to equipment, first confirm that Python can communicate
  with the device.

The example uses a simulated instrument. It requires no physical hardware,
graphical interface, optional Python package, or output file. It prints values
in the terminal and stops automatically after four seconds.

Define the InOut
----------------

:download:`Download the complete script
</downloads/custom_objects/custom_inout_regular.py>`, or create a file named
``custom_inout.py``. Its custom InOut is:

.. literalinclude:: /downloads/custom_objects/custom_inout_regular.py
   :language: python
   :start-after: # [custom-inout-class-start]
   :end-before: # [custom-inout-class-end]

The methods separate the device operations:

- ``__init__()`` accepts settings and stores values without contacting the
  device.
- ``open()`` prepares the device and its connection.
- ``get_data()`` acquires one sample. The first returned value is its
  timestamp, the remaining values are measurements.
- ``set_cmd()`` applies values received under the IOBlock's command labels.
- ``close()`` restores a safe state and releases the connection.

An input-only device needs ``get_data()`` but not ``set_cmd()``. An
output-only device needs ``set_cmd()`` but not ``get_data()``. When returning
a tuple, keep the number and order of values consistent. An InOut may instead
return a dictionary, which must contain the timestamp under ``t(s)``.

Use it with IOBlock
-------------------

The complete script configures the custom class through an IOBlock:

.. literalinclude:: /downloads/custom_objects/custom_inout_regular.py
   :language: python
   :start-after: # [custom-inout-use-start]
   :end-before: # [custom-inout-use-end]

The ``name`` value matches the ``SimulatedInOut`` class name. The two
``labels`` match the values returned by ``get_data()``. Values received under
``target_value`` are passed to ``set_cmd()``.

Before the test begins, ``make_zero_delay`` measures the simulated 1.5-unit
sensor offset and subtracts it from later readings. ``initial_cmd`` sets a
known starting command, while ``exit_cmd`` restores zero before ``close()`` is
called.

Run the simulation
------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_inout.py

After the brief zeroing step, the reported measurement alternates between
approximately +0.5 and -0.5. A final ``Generator Path exhausted`` warning
indicates the planned end of the example.

Adapt it to real equipment
--------------------------

.. warning::

   Before connecting an output-capable device, verify its command units,
   allowed range, power-on state, and independently safe shutdown method.
   Confirm that zeroing is meaningful for the selected channels. Software
   cleanup does not replace physical safeguards.

First validate connection, acquisition, commands, and cleanup in a small
independent Python program. Then move those operations into the matching
InOut methods:

1. Open and configure the connection in ``open()``, not ``__init__()``.
2. Timestamp each sample as close to the physical acquisition as practical.
3. Check command ranges before writing them in ``set_cmd()``.
4. Choose ``initial_cmd`` and ``exit_cmd`` values that are safe for the actual
   device, or omit them when sending such a command would be inappropriate.
5. Make ``close()`` safe after both a normal test and a partial setup failure.

The :doc:`test_hardware_object` tutorial shows how to check the resulting
InOut directly. See :class:`~crappy.inout.InOut` for the full custom interface
and :class:`~crappy.blocks.IOBlock` for every acquisition and command setting.
Chunked acquisition uses additional methods and is covered separately by the
:doc:`streaming_acquisition` tutorial.

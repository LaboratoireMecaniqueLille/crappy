.. _tutorial-custom-actuator:

========================
Create a custom Actuator
========================

This tutorial has one outcome: create a speed-controlled Actuator and use it
with a Machine Block.

Prerequisites
-------------

- Complete :doc:`actuator_control`.
- To adapt the example to equipment, first confirm that Python can connect to
  its controller and send the required commands.

The example uses a simulated linear stage. It requires no physical hardware,
graphical interface, optional Python package, or output file. It prints the
simulated speed and position in the terminal and stops automatically after
four seconds.

Define the Actuator
-------------------

:download:`Download the complete script
</downloads/custom_objects/custom_actuator_simulated.py>`, or create a file
named ``custom_actuator.py``. Its custom Actuator is:

.. literalinclude:: /downloads/custom_objects/custom_actuator_simulated.py
   :language: python
   :start-after: # [custom-actuator-class-start]
   :end-before: # [custom-actuator-class-end]

The methods have distinct jobs:

- ``__init__()`` accepts settings and stores values, but does not communicate
  with hardware.
- ``open()`` prepares the device. A real driver normally opens its connection
  and configures the controller here.
- ``set_speed()`` applies the command received by the Machine.
- ``get_speed()`` and ``get_position()`` return measurements for downstream
  Blocks.
- ``stop()`` brings the actuator to a safe stopped state.
- ``close()`` releases the device and its connection.

Only implement the command and measurement methods supported by the device.
For position control, define ``set_position(self, position, speed)``. The
``speed`` argument is always present but can be ``None`` when no target speed
was configured.

Control position-mode speed
---------------------------

In position mode, a Machine can provide that second argument in two ways:

- The ``speed`` entry in the Actuator configuration sets a fixed initial
  target speed.
- The ``speed_cmd_label`` entry names a label carrying updated target speeds.
  Each received value replaces the previous target.

If neither source has provided a value, Machine calls
``set_position(position, None)``. A controller that cannot adjust speed during
a position move may ignore the second argument, but its method must still
accept it.

Use it with Machine
-------------------

The complete script gives the class name and its settings to a Machine Block:

.. literalinclude:: /downloads/custom_objects/custom_actuator_simulated.py
   :language: python
   :start-after: # [custom-actuator-use-start]
   :end-before: # [custom-actuator-use-end]

The ``type`` value matches the ``SimulatedStage`` class name. The Machine uses
speed mode, forwards values from ``target_speed(mm/s)`` to ``set_speed()``,
and publishes the two measurements under the requested labels. The
``initial_position`` setting is passed to the class constructor.

Run the simulation
------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_actuator.py

The stage alternates between +2 mm/s and -2 mm/s. Its reported position first
increases and then decreases. A final ``Generator Path exhausted`` warning
indicates the planned end of the example. During cleanup, the Machine calls
``stop()`` and then ``close()``.

Adapt it to real equipment
--------------------------

.. warning::

   Before sending commands to machinery, verify the command units, allowed
   range, direction of motion, travel limits, and independent emergency-stop
   system. Begin with the load disconnected or the lowest safe command allowed
   by the equipment. Software cleanup does not replace physical safeguards.

Start from a small independent Python test of the manufacturer's library or
device protocol. Once connection, commands, readback, stopping, and cleanup
work reliably, move those operations into the matching Actuator methods:

1. Open and configure the connection in ``open()``, not ``__init__()``.
2. Validate command ranges before sending them in ``set_speed()`` or
   ``set_position()``.
3. Return only measurements the controller can actually provide.
4. Make ``stop()`` safe even after a partial setup or communication failure.
5. Make ``close()`` release the connection even if the test stopped early.

The :doc:`test_hardware_object` tutorial shows how to check the resulting
Actuator directly before placing it in a complete test. See
:class:`~crappy.actuator.meta_actuator.actuator.Actuator` for the full custom
interface and :class:`~crappy.blocks.Machine` for every Actuator setting.

.. _tutorial-feedback-loops:

==============================================
Control a simulated motor with a feedback loop
==============================================

This tutorial has one outcome: use a PID Block to make a simulated motor
follow a changing speed target.

Prerequisites
-------------

- Complete :doc:`command_generation` and :doc:`actuator_control`.
- Install Matplotlib:

  .. code-block:: shell-session

     python -m pip install matplotlib

- Run the example from a graphical desktop where Python can open a window.

The example uses ``FakeDCMotor`` and sends no command to physical hardware. It
opens one graph window, creates no file, and stops automatically after seven
seconds.

Create the feedback-control script
----------------------------------

:download:`Download the complete script
</downloads/more_complexity/feedback_loop.py>`, or create a file named
``feedback_loop.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/feedback_loop.py
   :language: python
   :start-after: # [feedback-loop-start]
   :end-before: # [feedback-loop-end]

The Generator publishes the requested motor speed as ``target_speed``. The
Machine drives the simulated motor with the ``voltage`` command and reports
its measured speed as ``actual_speed``.

The PID receives both speeds, compares them, and publishes the voltage needed
to reduce their difference. Its output is limited to the range from -10 to
10. The two Links below form the feedback loop:

- ``crappy.link(motor, controller)`` returns the measured result to the PID.
- ``crappy.link(controller, motor)`` sends the adjusted voltage to the motor.

The other Links provide the speed target to the PID and display both speeds.
The Generator's ``spam=True`` setting repeats the target so the graph receives
fresh points even while the target remains constant.

Run the simulated test
----------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python feedback_loop.py

The graph shows the target speed as steps and the measured speed approaching
each target. A final ``Generator Path exhausted`` warning indicates the planned
end of the example.

Adapt the loop to real equipment
--------------------------------

.. warning::

   Do not apply these example PID gains or voltage limits to physical
   equipment. First verify the actuator direction, command units, safe output
   range, travel limits, feedback signal, and emergency-stop system. Tune a
   controller from conservative limits with the load in a safe state.

First complete the open-loop checks in :ref:`the actuator-control tutorial
<tutorial-actuator-control>`. Replace ``FakeDCMotor`` and its simulation
arguments with the selected Actuator and its connection arguments. Then:

1. Keep the Machine's ``speed_label`` identical to the PID's ``input_label``.
2. Keep the PID's output label identical to the Machine's ``cmd_label``.
3. Set ``out_min`` and ``out_max`` to independently verified safe commands.
4. Tune ``kp``, ``ki``, and ``kd`` for the actual equipment.

See :class:`~crappy.blocks.PID` for all controller settings. Continue with
:doc:`modifiers` to transform selected data carried by a Link.

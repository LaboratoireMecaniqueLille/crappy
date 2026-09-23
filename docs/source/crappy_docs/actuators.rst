=========
Actuators
=========

Actuator drivers
----------------

Drivers whose documented path starts with ``crappy.collection`` are retained
for compatibility but are not actively maintained. Import
``crappy.collection`` before selecting one of them in a Machine Block.

Adafruit DC Motor Hat
+++++++++++++++++++++
.. autoclass:: crappy.collection.actuator.adafruit_dc_motor_hat.DCMotorHat
   :members: open, set_speed, stop, close
   :special-members: __init__

Fake DC Motor
+++++++++++++
.. autoclass:: crappy.actuator.FakeDCMotor
   :members: open, get_speed, get_position, set_speed, stop, close
   :special-members: __init__

Fake Stepper Motor
++++++++++++++++++
.. autoclass:: crappy.actuator.FakeStepperMotor
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close
   :special-members: __init__

JVL Mac140
++++++++++
.. autoclass:: crappy.actuator.JVLMac140
   :members: open, get_position, set_speed, set_position, reset_position,
             stop, close
   :special-members: __init__

Kollmorgen ServoStar 300
++++++++++++++++++++++++
.. autoclass:: crappy.actuator.ServoStar300
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

Newport TRA6PPD
+++++++++++++++
.. autoclass:: crappy.collection.actuator.newport_tra6ppd.NewportTRA6PPD
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

Oriental ARD-K
++++++++++++++
.. autoclass:: crappy.collection.actuator.oriental_ard_k.OrientalARDK
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Phidget Stepper4A
+++++++++++++++++
.. autoclass:: crappy.actuator.Phidget4AStepper
   :members: open, set_speed, set_position, get_speed, get_position, stop,
             close
   :special-members: __init__

Pololu Tic
++++++++++
.. autoclass:: crappy.actuator.PololuTic
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close
   :special-members: __init__

Schneider MDrive 23
+++++++++++++++++++
.. autoclass:: crappy.collection.actuator.schneider_mdrive_23.SchneiderMDrive23
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Parent Actuator
---------------

Actuator
++++++++
.. automodule:: crappy.actuator.meta_actuator.actuator

.. currentmodule:: crappy.actuator.meta_actuator.actuator

.. autoclass:: Actuator
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close, log
   :special-members: __init__

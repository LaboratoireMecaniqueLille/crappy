=========
Actuators
=========

Regular Actuators
-----------------

JVL Mac140
++++++++++
.. autoclass:: crappy.actuator.JVLMac140
   :members: open, get_position, set_speed, set_position, reset_position,
             stop, close
   :special-members: __init__

Schneider MDrive 23
+++++++++++++++++++
.. autoclass:: crappy.actuator.SchneiderMDrive23
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Fake Motor
++++++++++
.. autoclass:: crappy.actuator.FakeMotor
   :members: open, get_speed, get_position, set_speed, stop, close
   :special-members: __init__

Adafruit DC Motor Hat
+++++++++++++++++++++
.. autoclass:: crappy.actuator.DCMotorHat
   :members: open, set_speed, stop, close
   :special-members: __init__

Oriental ARD-K
++++++++++++++
.. autoclass:: crappy.actuator.OrientalARDK
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Pololu Tic
++++++++++
.. autoclass:: crappy.actuator.PololuTic
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close
   :special-members: __init__

Kollmorgen ServoStar 300
++++++++++++++++++++++++
.. autoclass:: crappy.actuator.ServoStar300
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

Newport TRA6PPD
+++++++++++++++
.. autoclass:: crappy.actuator.NewportTRA6PPD
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

FT232H Actuators
----------------

Adafruit DC Motor Hat FT232H
++++++++++++++++++++++++++++

.. autoclass:: crappy.actuator.DCMotorHatFT232H
   :members: open, set_speed, stop, close
   :special-members: __init__

Parent Actuator
---------------

Actuator
++++++++
.. autoclass:: crappy.actuator.Actuator
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close, log
   :special-members: __init__

Meta Actuator
+++++++++++++
.. autoclass:: crappy.actuator.MetaActuator
   :special-members: __init__

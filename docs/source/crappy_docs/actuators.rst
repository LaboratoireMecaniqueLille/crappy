=========
Actuators
=========

Regular Actuators
-----------------

Biaxe
+++++
.. autoclass:: crappy.actuator.Biaxe
   :members: open, set_speed, stop, close
   :special-members: __init__

Biotens
+++++++
.. autoclass:: crappy.actuator.Biotens
   :members: open, get_position, set_speed, set_position, reset_position,
             stop, close
   :special-members: __init__

CM Drive
++++++++
.. autoclass:: crappy.actuator.CMDrive
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Fake Motor
++++++++++
.. autoclass:: crappy.actuator.FakeMotor
   :members: open, get_speed, get_position, set_speed, stop, close
   :special-members: __init__

Motor kit pump
++++++++++++++
.. autoclass:: crappy.actuator.MotorKitPump
   :members: open, set_speed, stop, close
   :special-members: __init__

Oriental
++++++++
.. autoclass:: crappy.actuator.Oriental
   :members: open, get_position, set_speed, set_position, stop, close
   :special-members: __init__

Pololu Tic
++++++++++
.. autoclass:: crappy.actuator.PololuTic
   :members: open, get_speed, get_position, set_speed, set_position, stop,
             close
   :special-members: __init__

Servostar
+++++++++
.. autoclass:: crappy.actuator.Servostar
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

TRA6PPD
+++++++
.. autoclass:: crappy.actuator.TRA6PPD
   :members: open, get_position, set_position, stop, close
   :special-members: __init__

FT232H Actuators
----------------

Motor kit pump FT232H
+++++++++++++++++++++

.. autoclass:: crappy.actuator.MotorKitPumpFT232H
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

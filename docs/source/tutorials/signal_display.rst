.. _tutorial-signal-display:

========================
Display a signal live
========================

This tutorial has one outcome: display a measurement against time with a
:class:`~crappy.blocks.Grapher`.

Prerequisites
-------------

- Complete :doc:`quickstart` or be familiar with creating Blocks and Links.
- Install PyQtGraph and a supported Qt binding:

  .. code-block:: shell-session

     python -m pip install pyqtgraph PyQt6

- Run the example from a graphical desktop where Python can open a window.

The example uses a simulated machine and sends no command to physical
hardware. It opens one graph window, creates no file, and stops automatically
after five seconds.

Create the display script
-------------------------

:download:`Download the complete script
</downloads/getting_started/signal_display.py>`, or create a file named
``signal_display.py`` containing this code:

.. literalinclude:: /downloads/getting_started/signal_display.py
   :language: python
   :start-after: # [signal-display-start]
   :end-before: # [signal-display-end]

The Generator sends a constant speed command to ``FakeMachine``. The simulated
machine publishes several measurements, but the Grapher selects only ``t(s)``
for the horizontal axis and ``F(N)`` for the vertical axis. A Grapher can draw
several curves by receiving more label pairs.

Run the test
------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python signal_display.py

A window plots the simulated force as the test progresses. The window closes
when the Generator reaches the end of its five-second command. The final
``Generator Path exhausted`` warning indicates this planned end, not a
failure.

Display another signal
----------------------

Change the label pair given to the Grapher to select different axes. For
example, use ``('x(mm)', 'F(N)')`` to plot force against position:

.. code-block:: python

   graph = crappy.blocks.Grapher(('x(mm)', 'F(N)'))

Both labels must arrive together from the same upstream Block. Check that
Block's API entry for its output labels, or read
:doc:`../concepts/blocks_links_labels` for the general label rules.

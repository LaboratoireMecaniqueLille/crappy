======================
Hardware compatibility
======================

This page lists the hardware integrations distributed with Crappy. Inclusion
in the table means that a driver is available, it does not by itself claim
that every compatible device and operating-system combination was recently
tested. Follow a driver's name for its complete API and configuration details.
The operating-system column records intended or required platforms from the
implementation and existing documentation, it is not a verification claim.

How to read the status
----------------------

**Core — maintained** means that the driver is part of Crappy's public API and
that maintenance requests are accepted. **Collection — legacy** means that the
driver is retained for compatibility, is not actively maintained, and requires
an explicit ``import crappy.collection`` call in scripts using it.

Hardware verification is recorded separately:

- **Verified** means that the named configuration was tested on the displayed
  date.
- **Unverified** means that there is no sufficiently recent dated test record.
- **Software-only** identifies simulated or file-backed objects that do not
  communicate with physical hardware.

Camera integrations
-------------------

.. hardware-matrix:: Camera

InOut integrations
------------------

.. hardware-matrix:: InOut

Actuator integrations
---------------------

.. hardware-matrix:: Actuator

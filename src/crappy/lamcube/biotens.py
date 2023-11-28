# coding: utf-8

"""The name "Biotens" has been used for years before being replaced by
"JVLMac140". Keeping an alias here to make user's life easier."""

from ..actuator import JVLMac140


class Biotens(JVLMac140):
  """Simply an alias of :class:`~crappy.actuator.JVLMac140` for convenience for
   the users at the LaMcube laboratory."""

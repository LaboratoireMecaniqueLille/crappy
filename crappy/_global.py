# coding:utf-8

class OptionalModule:
  """Placeholder for optional dependencies when not installed

  Will print a message and raise an error when trying to use them
  """

  def __init__(self, module_name, message=None):
    self.mname = module_name
    if message is not None:
      self.message = message
    else:
      self.message = """The module {} is necessary to use this functionality
Please install it and try again""".format(self.mname)

  def __getattr__(self, _):
    print("Missing module: {}".format(self.mname))
    print(self.message)
    raise RuntimeError(self.message)

  def __call__(self, *_, **__):
    print("Missing module: {}".format(self.mname))
    print(self.message)
    raise RuntimeError(self.message)


class CrappyStop(Exception):
  """Error to raise when Crappy is terminating"""

  pass


class DefinitionError(Exception):
    """Error to raise when classes are not defined correctly"""

    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg

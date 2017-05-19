#coding:utf-8


class NotInstalled():
  def __init__(self,name):
    self.name = name

  def __getattr__(self,arg):
    m = """Module {} is not available,
make sure all dependencies are met
and try to reinstall Crappy""".format(self.name)
    raise NotImplementedError(m)

class NotSupported():
  def __init__(self,name):
    self.name = name

  def __getattr__(self,arg):
    m = "Module {} is not available on this platform".format(self.name)
    raise NotImplementedError(m)

class CrappyStop(Exception):
  """Error to raise when Crappy is terminating"""
  pass

class DefinitionError(Exception):
    """Error to raise when classes are not defined correctly"""
    def __init__(self,msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg


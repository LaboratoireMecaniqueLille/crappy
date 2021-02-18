import crappy
from time import time

speed = 2/60 # mm/sec


class Fake_machine(crappy.blocks.MasterBlock):
  def __init__(self,k,l0,fmax,cmd_label='cmd'):
    crappy.blocks.MasterBlock.__init__(self)
    self.freq = 100
    self.k = k
    self.l0 = l0
    self.fmax = fmax
    self.cmd_label = cmd_label
    self.pos = 0
    self.last_t = None

  def send_all(self):
    tosend = {
        't(s)':time()-self.t0,
        'F(N)':self.pos/self.l0*self.k,
        'x(mm)':self.pos,
        'Exx(%)':self.pos*100/self.l0
      }
    self.send(tosend)

  def begin(self):
    self.last_t = self.t0
    self.send_all()

  def loop(self):
    cmd = self.get_last()['cmd']
    t = time()
    dt = t-self.last_t
    self.pos += dt*cmd
    if self.pos/self.l0*self.k > self.fmax:
      self.k = 0
    self.send_all()
    self.last_t = t


generator = crappy.blocks.Generator(path=sum([
  [{'type':'constant','value':speed,'condition':'Exx(%)>{}'.format(i/10)},
  {'type':'constant','value':-speed,'condition':'F(N)<0'}]
  for i in range(10)], []),
spam=False)


machine = Fake_machine(k=210000*20*2,l0=200,fmax=30000)

crappy.link(generator,machine)
crappy.link(machine,generator)

graph_def = crappy.blocks.Grapher(('t(s)','Exx(%)'))
crappy.link(machine,graph_def)

graph_f = crappy.blocks.Grapher(('t(s)','F(N)'))
crappy.link(machine,graph_f)

graph_x = crappy.blocks.Grapher(('t(s)','x(mm)'))
crappy.link(machine,graph_x)

crappy.start()

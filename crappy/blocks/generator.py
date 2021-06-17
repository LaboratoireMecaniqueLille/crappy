# coding: utf-8

from time import time, sleep

from .block import Block
from . import generator_path
from .._global import CrappyStop


class Generator(Block):
  """This block is used to generate a signal.

  It can be used to drive a machine. This block can take inputs, and each path
  can use these inputs to take decisions.
  """

  def __init__(self,
               path=None,
               freq=200,
               cmd_label='cmd',
               cycle_label='cycle',
               cmd=0,
               repeat=False,
               trig_link=None,
               spam=False,
               verbose=False,
               end_delay=2):
    """Sets the args and initializes parent class.

    Args:
      path (:obj:`list`, optional): It must be a :obj:`list` of :obj:`dict`,
        each dict providing the parameters to generate the path. Each dict MUST
        have a key ``type``.

        Note:
          The Generator will then instantiate a :ref:`generator path` with all
          the other keys as `kwargs`, adding the current ``cmd`` and the time.

          On each round, it will call :meth:`Path.get_cmd` method, passing data
          until it raise :exc:`StopIteration`. It will then skip to the next
          path.

          When all paths are over, it will stop Crappy by raising
          :exc:`CrappyStop` unless ``repeat`` is set to :obj:`True`. If so, it
          will start over indefinitely.

      freq (:obj:`float`, optional): The frequency of the block. If set and
        positive, the generator will try to send the command at this frequency
        (in `Hz`). Else, it will go as fast as possible. It relies on the
        :ref:`Block` `freq` control scheme.
      cmd_label (:obj:`str`, optional): The label of the command to send in the
        links
      cycle_label (:obj:`str`, optional):
      cmd (:obj:`float`, optional): The first value of the command.
      repeat (:obj:`bool`, optional): Loop over the paths or stop when done ?
      trig_link (:obj:`str`, optional): If given, the block will wait until
        data is received through the input link with this label. If
        :obj:`None`, it will try loop at ``freq``.
      spam (:obj:`bool`, optional): If :obj:`True`, the value will be sent on
        each loop. Else, it will only send it if it was updated or we reached a
        new step.
      verbose (:obj:`bool`, optional): if :obj:`True`, displays a message when
        switching to the next path.
      end_delay (:obj:`float`, optional): The delay to wait for before raising
        the :exc:`CrappyStop` exception at the end of the path. This is meant
        to let enough time to the other blocks to properly terminate.
    """

    Block.__init__(self)
    self.niceness = -5
    self.freq = freq
    self.cmd_label = cmd_label
    self.cycle_label = cycle_label
    self.cmd = cmd
    self.repeat = repeat
    self.trig_link = trig_link
    self.spam = spam
    self.verbose = verbose
    self.end_delay = end_delay

    if path is None:
      path = []
    self.path = path
    assert all([hasattr(generator_path, d['type']) for d in self.path]), \
      "Invalid path in signal generator:" + \
      str(filter(lambda s: not hasattr(generator_path, s['type']), self.path))
    self.labels = ['t(s)', self.cmd_label, self.cycle_label]

  def prepare(self):
    self.path_id = -1  # Will be incremented to 0 on first next_path
    if self.trig_link is not None:
      self.to_get = list(range(len(self.inputs)))
      self.to_get.remove(self.trig_link)
    self.last_t = time()
    self.last_path = -1
    self.next_path()

  def next_path(self):
    self.path_id += 1
    if self.path_id >= len(self.path):
      if self.repeat:
        self.path_id = 0
      else:
        print("Signal generator terminated!")
        sleep(self.end_delay)
        # Block.stop_all()
        raise CrappyStop("Signal Generator terminated")
    if self.verbose:
      print("[Signal Generator] Next step({}):".format(self.path_id),
            self.path[self.path_id])
    kwargs = {'cmd': self.cmd, 'time': self.last_t}
    kwargs.update(self.path[self.path_id])
    del kwargs['type']
    name = self.path[self.path_id]['type'].capitalize()
    # Instantiating the new path class for the next step
    self.current_path = getattr(generator_path, name)(**kwargs)

  def begin(self):
    self.send([self.last_t - self.t0, self.cmd, self.path_id])
    self.current_path.t0 = self.t0

  def loop(self):
    if self.trig_link is not None:
      da = self.inputs[self.trig_link].recv_chunk()
      data = self.get_all_last(self.to_get)
      data.update(da)
    else:
      data = self.get_all_last()
    data[self.cmd_label] = [self.cmd]  # Add my own cmd to the dict
    try:
      cmd = self.current_path.get_cmd(data)
    except StopIteration:
      self.next_path()
      return
    # If next_path returns None, do not update cmd
    if cmd is not None and cmd is not self.cmd:
      self.cmd = cmd
      self.send([self.last_t - self.t0, self.cmd, self.path_id])
      self.last_path = self.path_id
    elif self.last_path != self.path_id:
      self.send([self.last_t - self.t0, self.cmd, self.path_id])
      self.last_path = self.path_id
    elif self.spam:
      self.send([self.last_t - self.t0, self.cmd, self.path_id])
    self.last_t = time()

# coding: utf-8

import numpy as np
import time
import pandas as pd
import os
import sys
from collections import OrderedDict

from .masterblock import MasterBlock

class SignalGenerator(MasterBlock):
  """
  WARNING: deprecated! Use Generator instead
  Generate a signal.
  """

  def __init__(self, path=None, **kwargs):
    """
    Calculate a signal, based on the time (from t0).

    ** DEPRECATED ** use Generator instead
    As t0 is used for evaluating the signal, multiple instances of this block will
    be synchronised.

    Args:
        path : list of dict
            Each dict must contain parameters for one step. See Examples section below.
            Available parameters are :

            * waveform : {'sinus','square','triangle','limit','hold'}
                Shape of your signal, for every step.
            * freq : int or float
                Frequency of your signal.
            * time : int or float or None
                Time before change of step, for every step. If None, means infinite.
            * cycles : int or float or None (default)
                Number of cycles before change of step, for every step. If None, means infinite.
            * amplitude : int or float
                Amplitude of your signal.
            * offset: int or float
                Offset of your signal.
            * phase: int or float
                Phase of your signal (in radians). If waveform='limit', phase will be
                the direction of the signal (up or down).
            * lower_limit : [int or float,sting]
                Only for 'limit' mode. Define the lower limit as a value of the
                labeled signal : [value,'label']
            * upper_limit : [int or float,sting]
                Only for 'limit' mode. Define the upper limit as a value of the
                labeled signal : [value,'label']
        send_freq : int or float , default = 800
            Loop frequency. Use this parameter to avoid over-use of processor and avoid
            filling the link too fast.
        repeat : Boolean, default=False
            Set True is you want to repeat your sequence forever.
        labels : list of strings, default =['t(s)','signal','cycle']
            Allows you to set the labels of output data.

    Returns:
        dict :
            time : float
                Relative time to t0.
            signal : float
                Generated signal. If waveform='limit', signal can be -1/0/1.
            cycle number : float
                Number of the current cycle.

    Examples:
        \code
        SignalGenerator(path=[{'waveform':'hold','time':3},
        {'waveform':'sinus','time':10,'phase':0,'amplitude':2,'offset':0.5,'freq':2.5},
        {'waveform':'triangle','time':10,'phase':np.pi,'amplitude':2,'offset':0.5,'freq':2.5},
        {'waveform':'square','time':10,'phase':0,'amplitude':2,'offset':0.5,'freq':2.5}
        {'waveform':'limit','cycles':3,'phase':0,'lower_limit':[-3,'signal'],'upper_limit':[2,'signal']}
        {"waveform":"protection","gain":1,"lower_limit":[-1,'F2(N)'],"upper_limit":[1,'F2(N)']}],
        send_freq=400,repeat=True,labels=['t(s)','signal'])
        \endcode

    In this example we displayed every possibility or waveform.
    Every dict contains informations for one step.
    The required informations depend on the type of waveform you need.
    """
    MasterBlock.__init__(self)
    print("WARNING! The SignalGenerator block is deprecated"\
        "Please use the Generator instead")
    self.path = path
    self.nb_step = len(path)
    self.send_freq = kwargs.get('send_freq', 800)
    self.repeat = kwargs.get('repeat', True)
    self.labels = kwargs.get('labels', ['time(sec)', 'cmd', 'cycle'])
    self.step = 0

  def main(self):
    try:
      last_t = self.t0
      cycle = 0
      first_of_step = True
      t_step = self.t0
      while self.step < self.nb_step:
        current_step = self.path[self.step]
        print("current step : ", self.step)
        try:
          self.waveform = current_step["waveform"]
          if self.waveform == 'hold':
            self.time = current_step["time"]
          elif self.waveform == 'limit':
            self.gain = current_step["gain"]
            self.cycles = current_step["cycles"]
            self.phase = current_step["phase"]
            self.lower_limit = current_step["lower_limit"]
            self.upper_limit = current_step["upper_limit"]
          elif self.waveform == 'ramp':
            self.gain = current_step["gain"]
            self.cycles = current_step["cycles"]
            self.phase = current_step["phase"]
            self.lower_limit = current_step["lower_limit"]
            self.upper_limit = current_step["upper_limit"]
            # This will allow continuous ramps if origin is not specified
            # However, it must be given for the first ramp
            # For now, continuity is guaranteed only between ramps!
            if 'origin' in current_step or self.step == 0:
              self.origin = current_step['origin']
          elif self.waveform == 'goto':
            self.direction = current_step["direction"]
            self.value = current_step["value"]
            self.offset = current_step["offset"]
          elif self.waveform == 'protection':
            self.gain = current_step["gain"]
            self.lower_limit = current_step["lower_limit"]
            self.upper_limit = current_step["upper_limit"]
          else:
            self.time = current_step["time"]
            self.phase = current_step["phase"]
            self.amplitude = current_step["amplitude"]
            self.offset = current_step["offset"]
            self.freq = current_step["freq"]

        except KeyError as e:
          print("You didn't define parameter %s for step number %s" % (e, self.step))
          raise

        if self.waveform == "goto":  # signal defined by a lower and upper limit
          cycle = 0
          while cycle == 0:
            timer = time.time()
            while timer - last_t < 1. / self.send_freq:
              time.sleep(-(timer-last_t - 1. / (self.send_freq))/10.)
              timer = time.time()
            last_t = time.time()
            recv = self.get_last()
            Data = pd.DataFrame([list(recv.values())],columns=list(recv.keys()))
            last_upper = (Data[self.value[1]]).last_valid_index()
            last_lower = (Data[self.value[1]]).last_valid_index()
            first_lower = (Data[self.value[1]]).first_valid_index()
            first_upper = (Data[self.value[1]]).first_valid_index()
            alpha = self.direction
            if abs(Data[self.value[1]][last_upper] - self.value[0]) < self.offset:  # if value > high_limit
              alpha = 0
              cycle = 1
            if last_upper != first_upper and last_lower != first_lower:  # clean old data
              Data = Data[min(last_upper, last_lower):]
            Array = OrderedDict(list(zip(self.labels, [last_t - self.t0, alpha, cycle])))
            self.send(Array)
          self.step += 1
          first_of_step = True
          cycle = 0
          t_step = time.time()
        elif self.waveform == "limit":  # signal defined by a lower and upper limit
          alpha = np.sign(np.cos(self.phase))
          while self.cycles is None or cycle < self.cycles:
            timer = time.time()
            while timer - last_t < 1. / self.send_freq:
              time.sleep(-(timer-last_t - 1. / (self.send_freq))/10.)
              timer = time.time()
            last_t = time.time()
            recv = self.get_last()
            Data = pd.DataFrame([list(recv.values())],columns=list(recv.keys()))

            last_upper = (Data[self.upper_limit[1]]).last_valid_index()
            last_lower = (Data[self.lower_limit[1]]).last_valid_index()
            first_lower = (Data[self.lower_limit[1]]).first_valid_index()
            first_upper = (Data[self.upper_limit[1]]).first_valid_index()
            if first_of_step:
              if alpha > 0:
                if Data[self.upper_limit[1]][last_upper] > self.upper_limit[0]:  # if value > high_limit
                  alpha = -1
              elif alpha < 0:
                if Data[self.lower_limit[1]][last_lower] < self.lower_limit[0]:  # if value < low_limit
                  alpha = 1
              first_of_step = False

            if self.upper_limit == self.lower_limit:  # if same limits
              alpha = 0
              cycle = time.time() - t_step
            if alpha > 0:
              if Data[self.upper_limit[1]][last_upper] > self.upper_limit[0]:  # if value > high_limit
                alpha = -1
                cycle += 0.5
            elif alpha < 0:
              if Data[self.lower_limit[1]][last_lower] < self.lower_limit[0]:  # if value < low_limit
                alpha = 1
                cycle += 0.5
            if last_upper != first_upper and last_lower != first_lower:  # clean old data
              Data = Data[min(last_upper, last_lower):t_data]
            Array = OrderedDict(list(zip(self.labels, [last_t - self.t0, alpha * self.gain, cycle])))
            self.send(Array)
          self.step += 1
          first_of_step = True
          cycle = 0
          t_step = time.time()
        elif self.waveform == "ramp":  # signal defined by a lower and upper limit
          alpha = np.sign(np.cos(self.phase))
          t_cycle = time.time()
          while self.cycles is None or cycle < self.cycles:
            timer = time.time()
            recv = self.get_last()
            while timer - last_t < 1. / self.send_freq:
              time.sleep(-(timer-last_t - 1. / (self.send_freq))/10.)
              timer = time.time()
            last_t = timer
            Data = pd.DataFrame([list(recv.values())],columns=list(recv.keys()))

            last_upper = (Data[self.upper_limit[1]]).last_valid_index()
            last_lower = (Data[self.lower_limit[1]]).last_valid_index()
            first_lower = (Data[self.lower_limit[1]]).first_valid_index()
            first_upper = (Data[self.upper_limit[1]]).first_valid_index()
            if first_of_step:
              if alpha > 0:
                if Data[self.upper_limit[1]][last_upper] > self.upper_limit[0]:  # if value > high_limit
                  alpha = -1
              elif alpha < 0:
                if Data[self.lower_limit[1]][last_lower] < self.lower_limit[0]:  # if value < low_limit
                  alpha = 1
              first_of_step = False

            if self.upper_limit == self.lower_limit:  # if same limits
              alpha = 0
              cycle = time.time() - t_step
            if alpha > 0:
              if Data[self.upper_limit[1]][last_upper] > self.upper_limit[0]:  # if value > high_limit
                self.origin = alpha * self.gain * (last_t-t_cycle)+self.origin
                alpha = -1
                cycle += 0.5
                t_cycle = time.time()
            elif alpha < 0:
              if Data[self.lower_limit[1]][last_lower] < self.lower_limit[0]:  # if value < low_limit
                self.origin = alpha * self.gain * (last_t-t_cycle)+self.origin
                alpha = 1
                cycle += 0.5
                t_cycle = time.time()
            if last_upper != first_upper and last_lower != first_lower:  # clean old data
              Data = Data[min(last_upper, last_lower):]
            Array = OrderedDict(list(zip(self.labels, [last_t - self.t0, alpha * self.gain * (last_t-t_cycle)+self.origin, cycle])))
            self.send(Array)
          self.step += 1
          first_of_step = True
          cycle = 0
          t_step = time.time()
        elif self.waveform == "protection":  # signal defined by a lower and upper limit
          while True:
            timer = time.time()
            while timer - last_t < 1. / self.send_freq:
              time.sleep(-(timer-last_t - 1. / (self.send_freq))/10.)
              timer = time.time()
            last_t = time.time()
            recv = self.get_last()
            Data = pd.DataFrame([list(recv.values())],columns=list(recv.keys()))

            last_t = time.time()
            last_upper = (Data[self.upper_limit[1]]).last_valid_index()
            last_lower = (Data[self.lower_limit[1]]).last_valid_index()
            first_lower = (Data[self.lower_limit[1]]).first_valid_index()
            first_upper = (Data[self.upper_limit[1]]).first_valid_index()

            if Data[self.upper_limit[1]][last_upper] > self.upper_limit[0]:  # if value > high_limit
              alpha = -1
            elif Data[self.lower_limit[1]][last_lower] < self.lower_limit[0]:  # if value < low_limit
              alpha = 1
            else:  # if in between the values
              alpha = 0
            cycle = -1
            if last_upper != first_upper and last_lower != first_lower:  # clean old data
              Data = Data[min(last_upper, last_lower):]
            # Array=pd.DataFrame([[last_t-self.t0,alpha*self.gain,cycle]],columns=self.labels)
            Array = OrderedDict(list(zip(self.labels, [last_t - self.t0, alpha * self.gain, cycle])))
            self.send(Array)
          self.step += 1
          first_of_step = True
          cycle = 0
          t_step = time.time()
        elif self.waveform == "hold":
          # print "holding"
          while self.time is None or (time.time() - t_step) < self.time:
            while time.time() - last_t < 1. / self.send_freq:
              self.get_last()
              # first=False
              time.sleep(1. / (100 * 1000 * self.send_freq))
            last_t = time.time()
            if self.step == 0:
              self.alpha = 0
            else:
              if self.path[self.step - 1]["waveform"] == "limit":
                self.alpha = 0
              elif self.path[self.step - 1]["waveform"] == "goto":
                self.alpha = 0
              elif self.path[self.step - 1]["waveform"] == "ramp":
                self.alpha = self.origin
              else:
                pass
            Array = OrderedDict(list(zip(self.labels, [last_t - self.t0, self.alpha, 0])))
            self.send(Array)
          self.step += 1
          first_of_step = True
          cycle = 0
          t_step = time.time()
        else:
          t_add = self.phase / (2 * np.pi * self.freq)
          while self.time is None or (time.time() - t_step) < self.time:
            while time.time() - last_t < 1. / self.send_freq:
              self.drop()
              time.sleep(1. / (100 * 1000 * self.send_freq))
            last_t = time.time()
            t = last_t + t_add
            if self.waveform == "sinus":
              self.alpha = self.amplitude * np.sin(2 * np.pi * (t - t_step) * self.freq) + self.offset
            elif self.waveform == "triangle":
              self.alpha = (4 * self.amplitude * self.freq) * (
                (t - t_step) - (np.floor(2 * self.freq * (t - t_step) + 0.5)) / (2 * self.freq)) * (-1) ** (
                np.floor(2 * self.freq * (t - t_step) + 0.5)) + self.offset
            elif self.waveform == "square":
              self.alpha = self.amplitude * np.sign(np.cos(2 * np.pi * (t - t_step) *
                                                           self.freq)) + self.offset
            else:
              raise Exception("invalid waveform : use sinus,triangle or square")
            cycle = 0.5 * np.floor(2 * ((t - t_step) * self.freq + 0.25))
            Array = OrderedDict(list(zip(self.labels, [t - self.t0, self.alpha, cycle])))
            self.send(Array)
          self.step += 1
          t_step = time.time()
          first_of_step = True
        if self.repeat and self.step == self.nb_step:
          self.step = 0
      raise Exception("Completed !")
    except (Exception, KeyboardInterrupt) as e:
      exc_type, exc_obj, tb = sys.exc_info()
      lineno = tb.tb_lineno
      print("Exception in SignalGenerator %s: %s line %s" % (os.getpid(), e, lineno))
      raise

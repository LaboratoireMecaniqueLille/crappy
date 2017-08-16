#coding: utf-8
from __future__ import print_function, division

import tables
import sys
import numpy as np
import matplotlib.pyplot as plt

NODE = "table" # Node containing the array to read
# If the range node is not specified, the digital levels (ints) will be returned
# If specified, it will use it to turn these levels in mV (SLOWER!)
RANGE_NODE = None # The name of the node storing the ranges of each channel
#RANGE_NODE = "ranges"

filename = None # Change here to ignore the prompt
start = 0 # Start to read from
stop = None # Where to stop (leave none to read the whole file)
step = None # Step (if None, will be computed and suggested)

if sys.version_info.major > 2:
  raw_input = input

if not filename:
  filename = raw_input("File name?")
h = tables.open_file(filename)
print(h)

arr = getattr(h.root,NODE)
lines,rows = arr.shape
print(lines,"lines and",rows,"rows")
a = rows*lines//500000 or 1
if a > 1:
  print("How many lines do you want to skip on each read?")
  print("1 will read ALL the data (may use too much memory!)")
  print("Suggested value:",a)
  if not step:
    step = int(raw_input("Read one out of how many lines?({})".format(a)) or a)
else:
  step = a

i = 0
if RANGE_NODE:
  out = np.empty((0,rows))
else:
  out = np.empty((0,rows),dtype=np.int16)
print("Reading...")
out = arr.read(start=start,stop=stop,step=step)
print(out.shape)

if RANGE_NODE:
  ranges = getattr(h.root,RANGE_NODE).read()
  print("Converting to mV...")
  h.close()
  for i,r in enumerate(ranges):
    out[:,i] *= r/32000
else:
  h.close()

print("Plotting...")
for i in range(rows):
  plt.plot(out[:,i])
print("Done!")
plt.show()

import crappy

SPEED = 5 # mm/min
FORCE_GAIN = 100 # N/V
POS_GAIN = 5 # mm/V

# This path will be a bit more complicated:
# Load until the strain is reached, unload until F(N)<1
paths = []
for strain in [.25,.5,.75,1.,1.5,2]:
  paths.append({'type':'constant',
    'value':SPEED/60,
    'condition':'Exx(%)>{}'.format(strain)}) # Go up to this level
  paths.append({'type':'constant',
    'value':-SPEED/60,
    'condition':'F(N)<1'}) # Go down to F=1N

# Just like the previous example, we create the generator with our new path
generator = crappy.blocks.Generator(paths,cmd_label='cmd')

# The DAQ board, this section is identical
force = {'name':'AIN0','gain':FORCE_GAIN}
pos = {'name':'AIN1', 'gain':POS_GAIN, 'make_zero':True}
cmd = {'name':'TDAC0','gain':POS_GAIN}
daq = crappy.blocks.IOBlock('Labjack_t7',channels=[force,pos,cmd],
    labels=['t(s)','F(N)','Position(mm)'],cmd_labels=['cmd'])

crappy.link(generator,daq)
# And we ALSO need to link the daq to the generator because it takes decisions
# based on the values of the force
# crappy.link(a,b) and crapp.link(b,a) are NOT equivalent because links are
# unidirectional ! Data only flows from the first block to the second
crappy.link(daq,generator)
# It is perfectly fine to make "loops" by linking in both directions like this

# Saver for the data from the machine
saver_daq = crappy.blocks.Saver('results_daq.csv')
crappy.link(daq,saver_daq)

# Now let's add our now sensor: videoextensometry
# We can specify arguments specific to the camera in the block
# save_folder asks the block to save the images
# It can be useful for further processing after the test
ve = crappy.blocks.Videoextenso('Webcam',width=1920,height=1080,
    save_folder='img/')
# When the program will start, this will open a windows to preview the
# video from the camera, ajust the settings and select the markers.

# Now we must link this block to the generator
crappy.link(ve,generator)
# and to a saver
saver_ve = crappy.blocks.Saver('results_ve.csv')
crappy.link(ve,saver_ve)
# Quick remark on savers: we used two separate savers for daq and ve. This is
# because savers can only take a SINGLE INPUT. Because the input blocks
# may run at different frequencies, it is not possible to build a csv file
# simply from two or more separate sources.
# We could interpolate them in the same
# timebase (see block Multiplex) but this means loosing the raw data. It is
# much better to perform this interpolation in post-processing if necessary

# Our force against time plot
graph_f = crappy.blocks.Grapher(('t(s)','F(N)'))
crappy.link(daq,graph_f)

# Let's add a second graph to plot strain against time
graph_s = crappy.blocks.Grapher(('t(s)','Exx(%)'))
crappy.link(ve,graph_s)

# We can now start the program
crappy.start()
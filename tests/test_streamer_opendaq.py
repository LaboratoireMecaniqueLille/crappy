from opendaq import DAQ
import time
dq = DAQ('/dev/ttyUSB0')
stream_exp = dq.create_stream(mode=0,
                              # 0:ANALOG_INPUT 1:ANALOG_OUTPUT 2:DIGITAL_INPUT 3:DIGITAL_OUTPUT 4:COUNTER_INPUT 5:CAPTURE_INPUT
                              period=1,
                              # 0:65536
                              npoints=0,
                              # 0:65536
                              continuous=True,
                              buffersize=1000)
stream_exp.analog_setup(pinput=2, ninput=5, gain=0, nsamples=254)

stream_exp2 = dq.create_stream(mode=0, period=1, npoints=0, continuous=True)
stream_exp2.analog_setup(pinput=4, gain=0, nsamples=254)
dq.start()

while True:
  try:
    # while dq.is_measuring():
    time.sleep(0.1)
    data = stream_exp.read()
    print 'data:', data[:5]
    print 'len(data)', len(data)
    data2 = stream_exp2.read()
    print 'data2:', data2[:5]
    print 'len(data2)', len(data2)
  except:
    dq.stop()
    break

# import crappy
#
# opendaq = crappy.technical.OpenDAQ()


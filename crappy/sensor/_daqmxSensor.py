# coding: utf-8
try:
	from PyDAQmx import *
except ImportError:
	print "WARNING : no module PyDAQmx installed, sensor won't work!"
import numpy
import time
from ._meta import acquisition

def getDAQmxDevicesNames():
    buffer_size = 4096
    buffer = ctypes.create_string_buffer(buffer_size)
    DAQmxGetSysDevNames(buffer, buffer_size)
    print len(buffer.value.split(",")), " devices detected: ", buffer.value
    return buffer.value.split(",")

class DaqmxSensor(acquisition.Acquisition):
    """Sensor class for Daqmx devices.""" 
    """PXI1Slot2, PXI1Slot3"""
    
    def __init__(self,device='Dev1',channels=0,
                 range_num=0, mode= "single"):
        
        self._ranges_tab = [[0.0,0.5],[0.0,1.0],[0.0,2.5],[0.0,5.0],[0.0,7.5],[0.0,10.0],
                     [-0.5,0.5],[-1.0,1.0],[-2.5,2.5],[-5.0,5.5],[-7.5,7.5],[-10.0,10.0]]
        self._channels_tab = ["ai0", "ai1","ai2","ai3","ai4","ai5","ai6","ai7", "ai8", "ai9", "ai10", "ai11", "ai12", "ai13", "ai14", "ai15"]
        
        # Declaration of variable passed by reference
        self.mode = mode
        self.channels=channels
        self.range_num=range_num
        
        
        #if type(self.channels)==list:
            ## if multiple channels
            #self.nchans=len(self.channels)
            #self.range_num=[self.range_num]*self.nchans
        #if type(self.gain)==int:
            #self.gain=[self.gain]*self.nchans
        #if type(self.offset)==int:
            #self.offset=[self.offset]*self.nchans
            #self.new()
        #else:
            #raise Exception("channels must be int or list")

        self.channel = self._channels_tab[channels]
        self.device = "%s/%s"%(device,self.channel)

    def new(self):
        DAQmxResetDevice(device)
        
        self.taskHandle = TaskHandle()
        self.read = int32()

        
        # DAQmx Configure Code
        DAQmxCreateTask("", byref(self.taskHandle))
        
        print self.device
        print type(self.device)

        buffer_size = 4096
        buffer = ctypes.create_string_buffer(buffer_size)
        DAQmxGetDeviceAttribute(device, DAQmx_Dev_ProductType, buffer)
        print buffer.value
        # if buffer.value == "PXIe-4331":
        #     print device
        #     DAQmxCreateAIBridgeChan(self.taskHandle, self.device,"",0.0,100.0,DAQmx_Val_VoltsPerVolt,DAQmx_Val_FullBridge,DAQmx_Val_Internal,1,20, None)
        # else:
        DAQmxCreateAIVoltageChan(self.taskHandle,self.device,"",DAQmx_Val_Cfg_Default,
                                 self._ranges_tab[self.range_num][0],self._ranges_tab[self.range_num][1],
                                 DAQmx_Val_Volts,None)

    def getData(self,nbPoints=1):
        """Read the signal"""
        try:
            DAQmxCfgSampClkTiming(self.taskHandle, "", 
                                  10000.0, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, 
                                  nbPoints+1)
            ## DAQmx Start Code
            DAQmxStartTask(self.taskHandle)
            
            data = numpy.zeros((nbPoints+1,), dtype=numpy.float64)
            t0 = time.time()
            DAQmxReadAnalogF64(self.taskHandle, nbPoints+1, 10.0, 
                               DAQmx_Val_GroupByChannel, data, 
                               nbPoints+1, byref(self.read), None) # DAQmx Read Code
            t=time.time()
            # DAQmx Stop Code
            DAQmxStopTask(self.taskHandle)
            delta = (t-t0)/float(nbPoints+1)
            print delta
            temps = [0]
            for x in xrange(1,nbPoints+1):
                temps.append(float(x)*delta)

            if(nbPoints==1):
                return (t, data[0])
            else:
                return (temps, data)
            
        except DAQError as err:
            raise Exception("DAQmx Error: %s"%err)


    def close(self):
        """Close the connection."""
        if self.taskHandle:
            # DAQmx Stop Code
            DAQmxStopTask(self.taskHandle)
            DAQmxClearTask(self.taskHandle)
        else:
            raise Exception('closing failed...')
    
    def __str__(self):
        return " Device: {0} \n Channels({1}): {2} \n Range({3}): min:{4} max: {5} \
               \n Mode: {6}".format(self.device,
                                    self.channels,self._channels_tab[self.channels], 
                                    self.range_num, self._ranges_tab[self.range_num][0],
                                    self._ranges_tab[self.range_num][1], self.mode)
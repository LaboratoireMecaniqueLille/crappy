import time
import numpy as np
import crappy
from datetime import datetime
from collections import OrderedDict

class CropCondition(crappy.links.MetaCondition):
    def __init__(self,cropped_size,offset=None):
        self.sliceY = slice(offset[0],cropped_size[0]+offset[0])
        self.sliceX = slice(offset[1],cropped_size[1]+offset[1])
        
    def evaluate(self, value):
        return value[self.sliceY, self.sliceX]

class AvgCondition(crappy.links.MetaCondition):
    def __init__(self,n=10):
        self.n = 10
        self.values = {}
        self.keys = []

    def evaluate(self,data):
        if self.keys == []:
            for k in data.keys():
                self.keys.append(k)
                self.values[k] = []
        r = OrderedDict()
        for k in self.keys:
            if len(self.values[k]) == self.n:
                del self.values[k][0]
            self.values[k].append(data[k])
            r[k] = sum(self.values[k])/len(self.values[k])
        return r

if __name__ == '__main__':
    try:
        d = datetime.now()
        ts="%02d-%02d-%04d_%02dh%02d"%(d.day,d.month,d.year,d.hour,d.minute) # Timestamp to append to the output files names
        savedir = "/media/biaxe/SSD1To/Victor/out/"+ts+"/"
        print "Timestamp de l'essai: "+ts
        ########################################### Creating objects
        # 5 sec of acquisition to average the force to init to the "real" 0
        F = [0,0,0,0] # The offsets
        FSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2, 3], gain=[10000,10000,10000,10000], offset=F)
        
        print "Initializing FSensors..."
        l = [[] for _ in range(4)]
        t0 = time.time()
        while time.time() < t0+5:
            data = FSensor.get_data('all')[1]
            for i in range(4):
                l[i].append(data[i])
        
        for i in range(4):
            F[i] = -sum(l[i])/len(l[i])


        FSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2, 3], gain=[10000,10000,10000,10000], offset=F)
        biaxeTech1 = crappy.technical.Oriental(port='/dev/ttyUSB0') # D
        biaxeTech2 = crappy.technical.Oriental(port='/dev/ttyUSB1') # A
        biaxeTech3 = crappy.technical.Oriental(port='/dev/ttyUSB2') # B
        biaxeTech4 = crappy.technical.Oriental(port='/dev/ttyUSB3') # C
        #biaxeTech1 = crappy.technical.Motion(motor_name='DummyTechnical')
        #biaxeTech2 = crappy.technical.Motion(motor_name='DummyTechnical')
        #biaxeTech3 = crappy.technical.Motion(motor_name='DummyTechnical')
        #biaxeTech4 = crappy.technical.Motion(motor_name='DummyTechnical')
        motors = [biaxeTech1, biaxeTech2, biaxeTech3, biaxeTech4]
        axes = []
        index = 1
        while len(motors) is not 0:
            for motor in motors:
                if motor.num_device == index:
                    axes.append(motor)
                    motors.remove(motor)
                    index+=1

    ########################################### Creating blocks
        
        x,y = 2048,2048
        compacter_effort=crappy.blocks.Compacter(20)
        camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=50, save=True,save_directory=savedir+"images/",xoffset=0, yoffset=0, width=x, height=y)
        
        mask = np.zeros((1024,1024),np.float32)
        mask[50:-50,50:-50] = 1
        
        correl = crappy.blocks.Correl((1024,1024),fields=['x','exx','y','eyy','exy','r','uxx','uxy','uyy','vxx','vxy','vyy'],verbose=2,levels=3,mask=mask)
        #correl = crappy.blocks.Correl((y,x),fields=['x','exx','y','eyy','exy','r'],verbose=2,levels=3,mask=mask,drop=False)
        compacter_correl=crappy.blocks.Compacter(20)
        graph_correl=crappy.blocks.Grapher(('t','exx'),('t','eyy'),('t','x'),('t','y'),('t','r'),('t','exy'),length=0)
        #graph_correl2=crappy.blocks.Grapher(('t','uxx'),('t','uxy'),('t','uyy'),('t','vxx'),('t','vxy'),('t','vyy'),length=0)
        
        
        save_effort=crappy.blocks.Saver(savedir+"test_correl_effort.txt")
        graph_effort=crappy.blocks.Grapher(('t(s)','F1(N)'),('t(s)','F2(N)'),('t(s)','F3(N)'),('t(s)','F4(N)'),length=0)
        
        
        effort=crappy.blocks.MeasureByStep(FSensor,labels=['t(s)','F1(N)','F2(N)','F3(N)','F4(N)'],freq=200)


        signalGenerator_1=crappy.blocks.SignalGenerator(path=[
                                {"waveform":"limit","gain":1,"cycles":1,"phase":0,"lower_limit":[3,'F1(N)'],"upper_limit":[1000,'F1(N)']}
                                ],
                                send_freq=5,repeat=False,labels=['t(s)','signal','cycle'])

        signalGenerator_2=crappy.blocks.SignalGenerator(path=[
                            {"waveform":"limit","gain":1,"cycles":1,"phase":0,"lower_limit":[0,'F2(N)'],"upper_limit":[1000,'F2(N)']}
                            ],
                            send_freq=5,repeat=False,labels=['t(s)','signal','cycle'])  ### Be careful with Eyy/Exx and motor axes !

        commandBiaxeX=crappy.blocks.CommandBiaxe(biaxe_technicals=[axes[1], axes[3]],speed=-3)
        commandBiaxeY=crappy.blocks.CommandBiaxe(biaxe_technicals=[axes[0], axes[2]],speed=-1)

        save_def=crappy.blocks.Saver(savedir+"test_correl_def.txt")
        display = crappy.blocks.CameraDisplayer(framerate=5)
        

    ########################################### Creating links
    
        lEff2Comp = crappy.links.Link(name='Eff2Comp',condition=AvgCondition(50))
        effort.add_output(lEff2Comp)
        compacter_effort.add_input(lEff2Comp)

        lComp2SaveF = crappy.links.Link(name='Comp2SaveF')
        compacter_effort.add_output(lComp2SaveF)
        save_effort.add_input(lComp2SaveF)
        
        lComp2GraphF = crappy.links.Link(name='Comp2GraphF')
        compacter_effort.add_output(lComp2GraphF)
        graph_effort.add_input(lComp2GraphF)
        
        lEff2Sig1 = crappy.links.Link(name='Eff2Sig1',condition=AvgCondition(50))
        effort.add_output(lEff2Sig1)
        signalGenerator_1.add_input(lEff2Sig1)
        
        lEff2Sig2 = crappy.links.Link(name='Eff2Sig2',condition=AvgCondition(50))
        effort.add_output(lEff2Sig2)
        signalGenerator_2.add_input(lEff2Sig2)
        
        lCorrel2Sig1 = crappy.links.Link(name='Correl2Sig1')
        correl.add_output(lCorrel2Sig1)
        signalGenerator_1.add_input(lCorrel2Sig1)
        
        lCorrel2Sig2 = crappy.links.Link(name='Correl2Sig2')
        correl.add_output(lCorrel2Sig2)
        signalGenerator_2.add_input(lCorrel2Sig2)
        
        lSig2BiaxeX = crappy.links.Link(name='Sig2BiaxeX')
        signalGenerator_1.add_output(lSig2BiaxeX)
        commandBiaxeX.add_input(lSig2BiaxeX)
        
        lSig2BiaxeY = crappy.links.Link(name='Sig2BiaxeY')
        signalGenerator_2.add_output(lSig2BiaxeY)
        commandBiaxeY.add_input(lSig2BiaxeY)
        

        lCam2Correl = crappy.links.Link(name='Cam2correl', condition=CropCondition((1024,1024),(512,512)))
        camera.add_output(lCam2Correl)
        correl.add_input(lCam2Correl)
        
        lCorrel2Comp = crappy.links.Link(name='Correl2Comp')
        correl.add_output(lCorrel2Comp)
        compacter_correl.add_input(lCorrel2Comp)
        
        lComp2GraphD = crappy.links.Link(name='Comp2GraphD')
        compacter_correl.add_output(lComp2GraphD)
        graph_correl.add_input(lComp2GraphD)
        
        #lComp2GraphD2 = crappy.links.Link(name='Comp2GraphD2')
        #compacter_correl.add_output(lComp2GraphD2)
        #graph_correl2.add_input(lComp2GraphD2)
        
        lComp2SaveD = crappy.links.Link(name='Comp2SaveD')
        compacter_correl.add_output(lComp2SaveD)
        save_def.add_input(lComp2SaveD)
        
        lCam2Disp = crappy.links.Link(name='Cam2Disp')
        camera.add_output(lCam2Disp)
        display.add_input(lCam2Disp)
        
        correl.init()
    ########################################### Starting objects

        t0=time.time()
        for instance in crappy.blocks._masterblock.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy.blocks._masterblock.MasterBlock.instances:
            instance.start()


    ########################################### Stopping objects

    except (Exception,KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy.blocks._masterblock.MasterBlock.instances:
            try:
                instance.stop()
                print "instance stopped : ", instance
            except:
                pass

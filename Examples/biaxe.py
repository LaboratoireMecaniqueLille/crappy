from __future__ import division
import crappy
import time

# conversion : 1 speed = 0.002 mm/s

def mean(l):
    return sum(l)/len(l)

class Mean(crappy.links.Condition):
    def __init__(self,nb_points=20):
        pass
        
        
    def evaluate(self,data):
        for k in data:
            print k
            val = data[k]
            nb = len(val)
            data[k] = [mean(val[:nb//2]),mean(val[nb//2:])]
        return data
 
def eval_offset(device, duration):
    """
    Method to evaluate offset. Returns an average of measurements made during specified time.
    """
    timeout = time.time() + duration  # secs from now
    print('Measuring offset (%d sec), please wait...'%duration)
    offsets = []


    while True:
        chan1 = device.get_data('all')[1]
        offsets.append(chan1)
        if time.time() > timeout:
            offsets = -np.mean(offset)
        print('offset:', offsets)
        break
    return offsets

if __name__ == '__main__':
    loading_path_list = []
    d = {"waveform": "limit", "gain": 1, "cycles": 1, "phase": 0,
                           "lower_limit": [0.1, 'F(N)']}

    for i in range(1,10):
      loading_path_list.append(d.copy())
      loading_path_list[-1].update({"upper_limit": [i*10,'Exx(%)']})

    loading_path_list.append({'waveform': 'hold', 'time': 5})
    timestamp = time.ctime()[:-5].replace(" ","_")
    savepath = "./"+timestamp+"/"
    Fsensor = crappy.sensor.ComediSensor(channels=[1], gain=[-3749.3], offset=[0])
    off1 = eval_offset(Fsensor,1)
    Fsensor = crappy.sensor.ComediSensor(channels=[1], gain=[-3749.3], offset=[-off1[0][0]])
    print "F:",Fsensor.get_data()

    biaxeTech1 = crappy.technical.Biaxe(port='/dev/ttyS4')
    biaxeTech2 = crappy.technical.Biaxe(port='/dev/ttyS5')

    # Creating blocks
    save_effort = crappy.blocks.Saver(savepath+"force.csv")
    graph_effort = crappy.blocks.Grapher(('t(s)', 'F(N)'),length=0)
    save_extenso = crappy.blocks.Saver(savepath+"extenso.csv")
    graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),length=0)

    effort = crappy.blocks.MeasureByStep(Fsensor, labels=['t(s)', 'F(N)'], freq=200, 
        white_spots=False,compacter=100)
    extenso = crappy.blocks.VideoExtenso(camera="XimeaCV", white_spot=False, display=True, compacter=3)

    signal_generator = crappy.blocks.SignalGenerator(
        path=loading_path_list,
        send_freq=100, repeat=False, labels=['t(s)', 'signal'])

    biaxe = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech1, biaxeTech2], speed=-1000)  # vertical

    crappy.link(effort,save_effort)
    crappy.link(effort,graph_effort)
    crappy.link(effort,signal_generator)

    crappy.link(extenso,save_extenso)
    crappy.link(extenso,graph_extenso)
    crappy.link(extenso,signal_generator)

    crappy.link(signal_generator,biaxe)


    crappy.start()

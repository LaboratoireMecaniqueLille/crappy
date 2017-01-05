# coding=utf-8
import time
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import pandas as pd
import crappy

crappy.blocks._masterblock.MasterBlock.instances = []  # Init masterblock instances


# class condition_def(
# # cette condition sert a remettre a 0 la def du videoextenso une fois passe un certain seuil d'effort
#     crappy.links.Condition):
#     def __init__(self, test=False):
#         self.FIFO = []
#         self.len_FIFO = 10.
#         self.start = False
#         self.first = True
#
#     def evaluate(self, value):
#         recv = self.external_trigger.recv(blocking=False)
#         if recv is not None:
#             self.start = True
#         if self.start and self.first:
#             if len(self.FIFO) < self.len_FIFO:
#                 self.FIFO.append([value['Lx'][0], value['Ly'][0]])
#             else:
#                 self.val0 = np.mean(self.FIFO, axis=0)  ###### check if axis is good
#                 self.first = False
#         elif not self.first:
#             value['Exx(%)'][0] = 100 * (value['Lx'][0] / self.val0[0] - 1)  # is that correct??
#             value['Eyy(%)'][0] = 100 * (value['Ly'][0] / self.val0[1] - 1)
#             return value
#         else:
#             return value


class ConditionOffset(crappy.links.Condition):
    def __init__(self):
        self.offset = 0.06  # a remplir avant de lancer l'essai
        self.first = True
        self.i = 0

    def evaluate(self, value):
        # print "f1 :", value
        if value['F(N)'][0] > self.offset and self.first:
            if self.i < 5:
                # print "f2"
                self.i += 1
                return None
            else:
                # print "f3"
                self.first = False
                return value
        else:
            # print "f4"
            return None


class ConditionDefPosition(crappy.links.Condition):
    def __init__(self):
        self.l0 = 7  # 7 mm au départ : je pense pas qu'on en ai besoin, c'est une def qu'on calcule??
        self.start = False

    def evaluate(self, value):
        # print "top1"
        # lecture non bloquante de l'effort : si on a pas dépassé l'offset en effort, aucune info n'a été envoyé
        recv = self.external_trigger.recv(blocking=False)
        # print "top2"
        if recv is not None:  # si on a reçu quelquechose:
            self.val0 = value['position'][
                0]  # on récupère la valeur de position au moment du assage de l'offset en effort
            self.start = True
        # print "top3"
        if self.start:
            value['Eyy(%)'] = pd.Series((100 * ((value['position'][0]) / self.val0 - 1)), index=value.index)
            # value['Eyy(%)'][0]=100*(value['Ly'][0]/self.val0[1]-1)
            # print "top4"
            return value
        else:
            # si ona rien reçu encore, envoie la value avec def=0 pour avoir
            # quand meme une def a envoyer au signalgenerator.
            value['Eyy(%)'] = pd.Series((0), index=value.index)  # def =0 tant qu'on a pas passé l'offset en effort

            # décommente ça si tu veux quand meme calculer la def au début, en prenant l0 = distance initiale (7 mm)
            # value['Exx(%)']=pd.Series((100*((value['position'][0])/self.l0-1)), index=value.index)
            # print "top5"
            return value


t0 = time.time()

try:
    # Creating objects

    instronSensor = crappy.sensor.ComediSensor(channels=[0], gain=[-48.8], offset=[0])
    t, F0 = instronSensor.get_data(0)
    print "offset=", F0
    instronSensor = crappy.sensor.ComediSensor(channels=[0], gain=[-48.8], offset=[-F0])
    biotensTech = crappy.technical.Biotens(port='/dev/ttyUSB0', size=30)

    # Creating blocks

    compacter_effort = crappy.blocks.Compacter(150)
    save_effort = crappy.blocks.Saver(
        "/home/biotens/Bureau/Annie/essais_rupture_3-11/E2124_rat10_LB_rupture_effort_1.txt")
    graph_effort = crappy.blocks.Grapher(('t(s)', 'F(N)'))

    compacter_extenso = crappy.blocks.Compacter(90)
    save_extenso = crappy.blocks.Saver(
        "/home/biotens/Bureau/Annie/essais_rupture_3-11/E2124_rat10_LB_rupture_extenso_1.txt")
    graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

    effort = crappy.blocks.MeasureComediByStep(instronSensor, labels=['t(s)', 'F(N)'], freq=150)
    extenso = crappy.blocks.VideoExtenso(camera="Ximea", white_spot=False,
                                          labels=['t(s)', 'Lx', 'Ly', 'Exx(%)', 'Eyy(%)'], display=True)
    # biotens=crappy.blocks.CommandBiotens(biotens_technicals=[biotensTech],speed=5)
    # signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"hold","time":0},
    # {"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.05,'F(N)'],"upper_limit":[90,'Eyy(%)']}],
    # send_freq=400,repeat=False,labels=['t(s)','signal'])
    # example of path:[{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[0.05,'F(N)'],
    # "upper_limit":[i,'Eyy(%)']} for i in range(10,90,10)]

    signalGenerator = crappy.blocks.SignalGenerator(path=[
        {"waveform": "limit", "gain": 1, "cycles": 3, "phase": 0, "lower_limit": [0.09, 'F(N)'],
         "upper_limit": [5, 'Eyy(%)']},
        # {"waveform":"limit","gain":1,"cycles":3,"phase":0,"lower_limit":[0.09,'F(N)'],"upper_limit":[10,'Eyy(%)']},
        # {"waveform":"limit","gain":1,"cycles":3,"phase":0,"lower_limit":[0.09,'F(N)'],"upper_limit":[25,'Eyy(%)']},
        {"waveform": "limit", "gain": 1, "cycles": 0.5, "phase": 0, "lower_limit": [0.09, 'F(N)'],
         "upper_limit": [90, 'F(N)']}],
        send_freq=5, repeat=False, labels=['t(s)', 'signal', 'cycle'])

    biotens = crappy.blocks.CommandBiotens(biotens_technicals=[biotensTech], speed=5)
    compacter_position = crappy.blocks.Compacter(5)
    save_position = crappy.blocks.Saver(
        "/home/biotens/Bureau/Annie/essais_rupture_3-11/E2124_rat10_LB_rupture_position_1.txt")
    graph_position = crappy.blocks.Grapher(('t(s)', 'Eyy(%)'))

    # Creating links

    link1 = crappy.links.Link()
    # link2=crappy.links.Link()
    link3 = crappy.links.Link()
    link4 = crappy.links.Link()
    link5 = crappy.links.Link()
    link6 = crappy.links.Link()
    link7 = crappy.links.Link()
    link8 = crappy.links.Link()
    link9 = crappy.links.Link()
    link10 = crappy.links.Link(ConditionDefPosition())
    link11 = crappy.links.Link()
    link112 = crappy.links.Link()
    link100 = crappy.links.Link(ConditionDefPosition())

    link12 = crappy.links.Link(ConditionOffset())
    link10.add_external_trigger(link12)
    link13 = crappy.links.Link(ConditionOffset())
    link100.add_external_trigger(link13)

    # Linking objects

    effort.add_output(link1)
    effort.add_output(link6)
    effort.add_output(link12)
    effort.add_output(link13)

    # extenso.add_output(link2)
    extenso.add_output(link3)

    signalGenerator.add_input(link1)
    signalGenerator.add_input(link100)  # j'envoie la valeur de la def mesurée par la position moteur
    signalGenerator.add_output(link9)

    biotens.add_input(link9)
    # c'est ici qu'on récupère la position du moteur non ? --> ici on recupere la position pour la sauvegarder
    biotens.add_output(link10)
    biotens.add_output(link100)  # on dédouble la sortie : ici on la récupere pour l'asservissement

    # on passe les 2 links 10 et 100 par la condition "condition_defposition" pour calculer la def et
    # la remttre a 0 quand l'effort depasse l'offset.

    compacter_effort.add_input(link6)
    compacter_effort.add_output(link7)
    compacter_effort.add_output(link8)

    save_effort.add_input(link7)

    graph_effort.add_input(link8)

    compacter_extenso.add_input(link3)
    compacter_extenso.add_output(link4)
    compacter_extenso.add_output(link5)

    save_extenso.add_input(link4)

    graph_extenso.add_input(link5)

    compacter_position.add_input(link10)
    compacter_position.add_output(link11)
    compacter_position.add_output(link112)

    save_position.add_input(link11)  # on sauvegarde la def moteur

    graph_position.add_input(link112)  # on représente la def moteur ???????

    # Starting objects

    t0 = time.time()
    for instance in crappy.blocks.MasterBlock.instances:
        instance.t0 = t0

    for instance in crappy.blocks.MasterBlock.instances:
        instance.start()

# Waiting for execution


# Stopping objects

except (Exception, KeyboardInterrupt) as e:
    print "Exception in main :", e
    # for instance in crappy.blocks._masterblock.MasterBlock.instances:
    # instance.join()
    for instance in crappy.blocks.MasterBlock.instances:
        try:
            instance.stop()
            print "instance stopped : ", instance
        except Exception as e:
            print e

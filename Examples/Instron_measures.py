#coding: utf-8
""" 
Base de travail pour l'acquisition et la commande des Instron à l'aide 
du LabJack T7. 
"""

import time
import crappy
import numpy as np

class EvalStress(crappy.links.Condition):
  """
  Classe utilisée pour effectuer des opérations sur les valeurs mesurées.
  Exemple avec un calcul de contraintes et de cycles.  
  """
  def __init__(self):
    
    self.surface = 10 * 7.  # Section de l'eprouvette (mm^2)
    # Variables utilisées pour la détection du nombre de cycles
    self.signe = 1  
    self.nb_cycles = 0.
    # Si la valeur moyenne du signal n'est pas centrée en zéro (ou pour 
    # filtrer le bruit), modifier la valeur ci-dessous
    self.mean = 0.
    
  def evaluate(self, value):
    
    value["Contrainte(MPa)"] = value["effort(kN)"] * 1000 / self.surface
    # On détecte la demi-période en surveillant les changements de signe de la
    # déformation.
    if value["Deformation(%)"] * self.signe - self.mean < 0:
      self.nb_cycles += 0.5
      self.signe *= -1
    value["Cycles"] = self.nb_cycles
    return value

# LABJACK: Acquisition sur voie AIN0, AIN1, AIN2, commande sur voie TDAC0
# AIN0: position
# AIN1: effort
# AIN2: Deformation
# TDAC0: Commande 

# Décommenter les lignes ci-dessous si la commande est utilisée
#labjack = crappy.inout.Labjack_t7(out_channels="TDAC0")
#labjack.open()
#labjack.set_cmd(0)
#labjack.close()

labjack = crappy.blocks.IOBlock("Labjack_T7",
                                labels=["temps(sec)", "Deplacement(mm)",
                                        "effort(kN)", "Deformation(%)"],
                                channels=["AIN0", "AIN1", "AIN2"],
                                gain=[0.5, 8, 0.2],  # V/mm; kN/V; V/%
                                offset=0,
                                chan_range=10,
                                make_zero=True,
                                resolution=0,
                                # Décommenter si usage de la commande
                                #out_channels='TDAC0',
                                #cmd_labels=["position_command"],
                                verbose=True)
"""
Décommenter ci-dessous si usage de la commande : il est possible de faire 
des chemins très compliqués, ici on montre l'exemple de 2 cycles avec des 
limites hautes et basses en déformation.
"""
#pente = 0.05 # fois l'amplitude = mm/s
#paliers = [0.2, 0.3]
##paliers = np.arange(0.2, 1.1, 0.1)
#nb_cycles = 2
#path = []
#for palier in paliers:
#  path.append(
#      {
#        "type": "cyclic_ramp",
#        "speed1": pente,
#        "condition1": "Deformation(%)>"+str(palier),
#        "speed2": -pente,
#        #"condition2": "Deformation(%)<0",
#        "condition2": "effort(kN)<0",
#        "cycles": nb_cycles
#      })
#  
#signal = crappy.blocks.Generator(path=path,
#                                cmd_label="position_command")
# Liaison du signal avec le labjack
#crappy.link(signal, labjack)
#crappy.link(labjack, signal)

grapher_effort = crappy.blocks.Grapher(("temps(sec)", "Contrainte(MPa)"),
                                       length=0)
grapher_position = crappy.blocks.Grapher(("temps(sec)", "Deformation(%)"),
                                         length=0)

save_folder = '/home/francois/Code/_Projets/16_test_cycles/' \
               + time.strftime("%y.%m.%d-%H:%M/")
saver = crappy.blocks.Saver(save_folder + 'Traction_1.csv')


crappy.link(labjack, saver, condition=EvalStress())
crappy.link(labjack, grapher_position)
crappy.link(labjack, grapher_effort, condition=EvalStress())
raw_input("Appuyer sur une touche pour lancer le programme...")
crappy.start()


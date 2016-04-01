# coding: utf-8
import numpy as np
from collections import OrderedDict

#xp = [1, 2, 3]
#fp = [3, 2, 0]
##b=np.interp(2.5, xp, fp)
##print b
#a=np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp) #T base de temps petite, t base de temps grande, points a vouloir interpoler ( Cycle, resistance, position)
#print a
##UNDEF = -99.0
##np.interp(3.14, xp, fp, right=UNDEF)

class Interpolation (object):
    def __init__(self):#Recuperation des donnees en entree.
        self.inputs[0].recv()=data
        
    def interp(self, 'var1','var2','var3','var4','var5','var6'): ## t(s) t1(s) cycle resistance(ohm) dep(mm) F(N)
        tpetit=self.data['var1'] # temps avec la plus petite base
        tgrand=self.data['var2'] # temps avec la plus grande base
        cycle=self.data['var3'] # variables a interpoler
        resistance=self.data['var4']
        position=self.data['var5']
        effort=self.data['var6']
        
        interp_cycle=np.interp('var1','var2','var3')
        interp_resistance=np.interp('var1','var2','var4')
        
        Array=OrderedDict(zip(['t(s)','cycle','resistance','position','effort'],[tpetit,interp_cycle,interp_resistance,position,effort]))
        
        try:
            
            for output in self.outputs:
                output.send(Array)
                                    
        except :
            print "Erreur envoi Array interpolation"
            

        
    
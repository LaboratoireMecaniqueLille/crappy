#import pandas as pd

#a=pd.DataFrame([[1,2,3]],columns=['a','b','c'])
#while True:
	#for i in range(800):
		#if i==0:
			#Data=a
		#else:
			#Data1=a
		#if i!=0:
			#Data=pd.concat([Data,Data1])

import pandas as pd
import numpy as np
from multiprocessing import Pipe
a,b=Pipe()
c=pd.DataFrame(np.random.rand(100,3),columns=['a','b','c'])

while True:
    a.send(c)
    b.recv()
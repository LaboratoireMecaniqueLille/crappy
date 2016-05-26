# This file provide a way to open the files saved by crappy videoxtenso. Please don't modify this file directly but make a copy for your own usage.
# Note that there is a special file in this folder for other data, as they have a specific format.

import numpy as np
import scipy.ndimage as nd
import scipy.optimize as optimization
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan', linewidth=500,suppress=True)

file_path='/home/corentin/Bureau/delete2.txt'
columns=(0,1,2,3,4)
if __name__ == '__main__':
	### basic commands to open a log file 
	def get_data(file_path,columns):
		a=np.loadtxt(file_path,dtype=str,usecols=(columns),skiprows=1,delimiter="' '",converters = {0: lambda s: float(s.split("'")[-1]),
																								1:lambda s: (s.strip().replace("  ",",").replace(" ","")),
																								2:lambda s: (s.replace("  ",",").replace(" ","")),
																								3:lambda s: float(s),
																								4:lambda s: float(s.split("'")[0])}) # load the file in str format ### put here the column you need ,converters = ({0: lambda s: float(s.lstrip("['")) or float(s.lstrip("[['"))})
		data=np.delete(a,[1,2],1).astype(np.float)
		
		Px_=a[:,1]
		Px1=np.char.replace(Px_,"[","")
		Px2=np.char.replace(Px1,"]","")
		Px3=np.char.split(Px2,",")
		Px4=[filter(None, Px3[i]) for i in range(len(Px3))]#remove blanks
		Px=np.array([np.array(Px4[i]).astype(np.float) for i in range(len(Px4))])
		
		Py_=a[:,1]
		Py1=np.char.replace(Py_,"[","")
		Py2=np.char.replace(Py1,"]","")
		Py3=np.char.split(Py2,",")
		Py4=[filter(None, Py3[i]) for i in range(len(Py3))] #remove blanks
		Py=np.array([np.array(Py4[i]).astype(np.float) for i in range(len(Py4))])


		return data,Px,Py

	#data2=get_data('/home/corentin/Bureau/_videoExtenso_to_delete.py',(1,2,3,4,5))
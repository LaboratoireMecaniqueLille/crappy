# This file provide a way to open the files saved by crappy. Please don't modify this file directly but make a copy for your own usage.
# Note that there is a special file in this folder for videoextenso data, as they have a specific format.

import numpy as np
import scipy.ndimage as nd
import scipy.optimize as optimization
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan', linewidth=500,suppress=True)


if __name__ == '__main__':
	### basic commands to open a log file 
	def get_data(file_path,columns):
		a=np.loadtxt(file_path,dtype=str,usecols=(columns),skiprows=1) # load the file in str format ### put here the column you need 
		b=np.char.rstrip(a, ']')  # remove the useless ]
		print b[0]
		c=b.astype(np.float64)  # convert to float
		#if list(c[-1,0])==0.:
			#d=c[0:(list(c[:,0])).index(0.00000000e+00),:]  # remove all the useless 0's at the end 
		data=np.transpose(c) # allow to use data as column
		return data

	#['t(s)', 'def(%)', 'dist(deg)', 'def_plast(%)', 'E(MPa)', 'G(Mpa)', 'status'] TTC
	#['t(s)', 'def(%)', 'F(N)', 'dist(deg)', 'C(Nm)', 'tau(Pa)', 'sigma(Pa)', 'eps_tot(%)'] data_instron
	#<<<<<<< HEAD
	data2=get_data('/home/corentin/Bureau/test_TTC.txt',(1,2,3,4,5,6,7))
	data=get_data('/home/corentin/Bureau/data_instron.txt',(1,2,3,4,5,6,7,8))
	#data=get_data('/home/essais-2015-3/Bureau/t_dep_F.txt',(1,2,3))

	##=======
	#data=get_data('/home/annie/Bureau/bordel/test_effort_extenso_1.txt',(1,2,3,4,5))
	#>>>>>>> 326d784433793d074bb6b045fb78f19fff800084
	
	#plt.plot(data[0],data[4],'r',label='E')
	#plt.plot(data[0],data[5],'b',label='G')
	#plt.grid()
	#plt.legend()
	#plt.figure()
	#plt.plot(data[0],data[3],'b',label='def_p(%)')
	#plt.legend()
	#plt.figure()
	#plt.plot(data[0],data[5],'b',label='status')
	#plt.legend()
	#plt.grid()
	total_stress=np.sqrt(data[3]**2+3*data[4]**2)
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(data[1],data[6],'bo',label='sigma')
	ax1.plot(data[3],data[5],'ro',label='tau')
	ax2.plot(data[1],data[2],'b',label='traction ')
	ax2.plot(data[3],data[4],'r',label='torsion ')
	#ax1.plot(data[0],total_stress,'g',label='total_stress')
	#ax2.plot(data[2],data[4],'k',label='total_strain')
	##ax2.plot(data[0],data[1],'y',label='def')
	##ax2.plot(data[0],data[2],'g',label='dist')
	#ax2.plot(data2[0],data2[3],'y',label='plast_def')
	#ax1.plot(data2[0],data2[4],label='E')
	#ax1.plot(data2[0],data2[5],label='G')
	plt.legend()
	plt.grid()
	#plt.figure()
	#plt.plot(data[5],data[3],'b',label='strain-stress traction')
	#plt.plot(data[5],data[4],'r',label='strain-stress torsion')
	#plt.legend()
	#plt.grid()


	#data_t=np.transpose(data)
	#data2=get_data('/home/corentin/Bureau/signal_adapted.txt',(1,2))

	#rmoy=(25+22)*10**(-3)/2
	#I=np.pi*((25*10**-3)**4-(22*10**-3)**4)/32
	#contrainte=data[2]/(110.74*10**(-6))
	#moment=(data[4]/I)*rmoy
	#moment_smooth=nd.gaussian_filter(moment,200)
	#contrainte_smooth=nd.gaussian_filter(contrainte,200)
	#fig, ax1 = plt.subplots()
	#ax1.plot(data[0][0::],data[2][0::],'b')
	#ax2 = ax1.twinx()
	#ax2.plot(data[0][0::],data[1][0::],'r')
	#ax1.set_ylabel('F(N)', color='b')
	#for tl in ax1.get_yticklabels():
		#tl.set_color('b')
    
	#ax2.set_ylabel('def(%)', color='r')
	#for tl in ax2.get_yticklabels():
		#tl.set_color('r')
	
	##fig=plt.figure()
	#fig, ax3 = plt.subplots()
	#ax4 = ax3.twinx()
	#ax4.plot(data[0][0::],data[3][0::],'r')
	
	#ax3.plot(data[0][0::],data[4][0::],'b')
	#ax3.set_ylabel('C(Nm)', color='b')
	#for tl in ax3.get_yticklabels():
		#tl.set_color('b')
	#ax4.set_ylabel('dist(deg)', color='r')
	#for tl in ax4.get_yticklabels():
		#tl.set_color('r')
	
	#fig, ax1 = plt.subplots()
	#ax1.plot(data[0][0::],contrainte_smooth,'b')
	#ax2 = ax1.twinx()
	#ax2.plot(data[0][0::],moment_smooth,'r')
	#ax1.set_ylabel('Sigma(MPa)', color='b')
	#for tl in ax1.get_yticklabels():
		#tl.set_color('b')
    
	#ax2.set_ylabel('Moment(MPa)', color='r')
	#for tl in ax2.get_yticklabels():
		#tl.set_color('r')
	
	
	#plt.figure()
	#plt.plot(contrainte_smooth,moment_smooth,'g')
	#plt.grid()
	#plt.plot(data[0],data[1],'+b')
	#plt.figure()
	#plt.plot(data[0],data[2],'+b')
	##plt.figure()
	#plt.plot(data2[0],data2[2],'+r')
	#plt.plot(data[0],data[2],'+r')
	#plt.plot(data2[0],data2[1],'+r')
	plt.show()
	#offset=36.662803889999999
	#for i in range (data_t.shape[0]):
	#data_t[i][0]-=offset
	#np.savetxt("save_extenso.txt",data_t)

	#data_to_write=""
	#for i in range data.shape[1]:
	#for j in range data.shape[0]:
		#data_to_write+= str(data[j][i])
	#data_to_write+="\n"


	#def func(x, a,b):
		#return  b*x+a
	
	#path="/home/biaxe/Code/Biaxe/Python_test/Results/"
	#for i in range(1,2):
	#file_name=path+str(i)+'.txt'
	#data=get_data(file_name,(2,3,4))
	#data=nd.median_filter(data,3)
	#xdata=data[0][2000:-2000]
	#y1data=data[1][2000:-2000]
	#y2data=data[2][2000:-2000]
	#x0    = np.array([0.0, 0.0])
	#P, pcov = optimization.curve_fit(func, xdata, y1data, x0)
	#print P, pcov
	##plt.plot(xdata,y2data)
	#plt.figure();
	#plt.plot(xdata,P[1]*xdata+P[0],xdata,y1data)
	#plt.figure()
	#plt.plot(P[1]*xdata+P[0]-y1data),plt.show()

	### -------------------------specific loop for calculating the mean value
	#mean=[]
	#path="/home/corentin/Bureau/Temperature_thermocouple/"
	#for i in range(0,18,1):
	#file_name='T'+str(i)+'.txt'
	#a=np.loadtxt(path+file_name,dtype=str,usecols=(1,2)) # load the file in str format
	#b=np.char.rstrip(a, ']')  # remove the useless ]
	#c=b.astype(np.float64)  # convert to float
	#d=c[0:(list(c[:,0])).index(0.00000000e+00),:]  # remove all the useless 0's at the end 
	#mean.append(np.mean(d[:,1]))
	
	
	###---------------------- specific loop for calculating the T values during traction
	#path="/home/corentin/Bureau/Temperature_thermocouple/"
	#d=[]
	#for i in range(13,17,1):
	#file_name='T'+str(i)+'-'+str(i+1)+'_traction'+'.txt'
	#a=np.loadtxt(path+file_name,dtype=str,usecols=(1,2)) # load the file in str format
	#b=np.char.rstrip(a, ']')  # remove the useless ]
	#c=b.astype(np.float64)  # convert to float
	#d.append(nd.gaussian_filter(c[:,1],10))  

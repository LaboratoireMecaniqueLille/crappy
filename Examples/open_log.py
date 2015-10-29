import numpy as np
import scipy.ndimage as nd
import scipy.optimize as optimization
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan', linewidth=500,suppress=True)

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


#data=get_data('/home/essais-2015-3/Bureau/tension_coeff.txt',(4,5))
data=get_data('/home/essais-2015-3/Bureau/data_fissuration/tension_coeff.txt',(4,5))


#data=get_data('/home/essais-2015-1/Bureau/t_dep_F.txt',(1,2,3))

#data_t=np.transpose(data)
#data2=get_data('/home/corentin/Bureau/signal_adapted.txt',(1,2))


#plt.plot(data[0][0::2],data[1][0::2],'+b')
##plt.figure()
#plt.plot(data[0][1::2],data[1][1::2],'+r')
#plt.plot(data[0],data[1],'+b')
##plt.figure()
plt.plot(data[0],data[1],'+r')
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
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica'],'size':22})
rc('text', usetex=True)


# exemple graphe essai traction
E=210.*10**9
k=100
x=np.linspace(0,0.001,100)
y=E*x
#x2=np.arange(0.25,5,0.5)
x2=np.linspace(0,100*10**6,1000)
a=9.752380952380953e-19
b=1/E
y2=a*x2**2+b*x2+x[-1]
x2+=210*10**6
X=np.concatenate((np.asarray(x),np.asarray(y2)))
Y=np.concatenate((np.asarray(y),np.asarray(x2)))
plt.plot(X,Y/10**6,'b',label='signal')
#plt.plot(y2,x2+210*10**6,'b+')
plt.xlim(0,0.01)
plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$\sigma~($MPa)")
#plt.legend(bbox_to_anchor=(0, 1.02, 1., .102),loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.grid()
plt.show()


# traction cyclique:
x=np.linspace(0,5,1000)
y=np.sin(2*np.pi*x)
x2=np.arange(0.25,5,0.5)
y2=np.sin(2*np.pi*x2)

plt.plot(x,y,label='signal')
plt.plot(x2,y2,'og',label="prise d\'images")
plt.ylim(-1.1,1.1)
plt.xlabel("time(s)")
plt.ylabel(r"$\varepsilon~($\%)")
#plt.legend(bbox_to_anchor=(0, 1.02, 1., .102),loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.grid()
plt.show()

# videoextenso
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x=np.linspace(0,5,21)
y=0.5*np.floor(np.cos(2*np.pi*x-np.pi/2)+1.1)
y[3::4]+=0.05
#x2=np.arange(0.25,5,0.5)
y2=np.ones_like(x)
y3=4.3*y2

lns1=ax1.plot(x,y,label='signal')
lns2=ax1.plot(x,y2,'r',label="limite haute")
lns3=ax2.plot(x,y3,'g',label="limite basse")
plt.ylim(0,1.1)
plt.xlabel("time(s)")
plt.ylabel(r"$\varepsilon~($\%)")
ax1.set_ylabel(r"$\varepsilon~($\%)", color='r')
ax1.set_ylim(0,1.1)
ax2.set_ylabel('F(N)', color='g')
ax2.set_ylim(0,100)
for tl in ax1.get_yticklabels():
	tl.set_color('r')
    
for tl in ax2.get_yticklabels():
	tl.set_color('g')
	
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]

#ax1.legend(lns, labs, bbox_to_anchor=(0, 1.02, 1., .102),loc=3, ncol=3, mode="expand", borderaxespad=0.)

ax1.grid()
plt.show()

# fissuration output:
x=np.linspace(0,5,1000)
y=np.sin(2*np.pi*x)
y2=y/(np.sqrt(1+x))
y3=1/(np.sqrt(1+x))
y4=-1/(np.sqrt(1+x))

plt.plot(x,y3,'b',label='signal')
plt.plot(x,y4,'b',label='signal')
plt.plot(x,y2,'r',label="F(N)")
plt.ylim(-1.1,1.1)
plt.xlabel("time(s)")
plt.ylabel("F")
#plt.legend(bbox_to_anchor=(0, 1.02, 1., .102),loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.grid()
plt.show()

#fissuration input
x=np.linspace(0,5,1000)
y=0.45*np.sin(2*np.pi*x)+0.55
x2=np.arange(0.25,5,0.5)
y2=np.sin(2*np.pi*x2)
x3=np.arange(0.75,5,1)
y3=0.45*np.sin(2*np.pi*x3)+0.55


plt.plot(x,y,label='signal')
plt.plot(x2,y2,'og',label="prise d\'images")
plt.plot(x3,y3,'or',label="prise d\'images")
plt.ylim(0,1.1)
plt.xlabel("time(s)")
plt.ylabel("F(N)")
#plt.legend(bbox_to_anchor=(0, 1.02, 1., .102),loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.grid()
plt.show()
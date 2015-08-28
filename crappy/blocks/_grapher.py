from _meta import MasterBlock
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd

class Grapher(MasterBlock):
	"""Plot the input data"""
	def __init__(self,mode,*args):
		"""
Grapher(mode,*args)

The grapher receive data from the Compacter (via a Link) and plots it.

Parameters
----------
mode : string
	"dynamic" : create a dynamic graphe that updates in real time. 
	"static" : create a graphe that add new values at every refresh. If there 
	is too many data (> 20000), delete one out of 2 to avoid memory overflow.
args : tuple
	tuples of the columns labels of input data for plotting. You can add as
	much as you want, depending on your computer performances.

Examples:
---------
graph=Grapher("dynamic",('t(s)','F(N)'),('t(s)','def(%)'))
	plot a dynamic graph with two lines plot( F=f(t) and def=f(t)
		"""
		print "grapher!"
		self.mode=mode
		self.args=args
		self.nbr_graphs=len(args)		
      
	def main(self):
		try:
			print "main grapher"
			if self.mode=="dynamic":
				save_number=0
				fig=plt.figure()
				ax=fig.add_subplot(111)
				for i in range(self.nbr_graphs):	# init lines
					if i ==0:
						li = ax.plot(np.arange(1),np.zeros(1))
					else:
						li.extend(ax.plot(np.arange(1),np.zeros(1)))
				plt.grid()
				fig.canvas.draw()	# draw and show it
				plt.show(block=False)
				while True:
					Data=self.inputs[0].recv()	# recv data
					legend_=Data.columns[1:]
					if save_number>0: # lose the first round of data    
						if save_number==1: # init
							var=Data
							plt.legend(legend_,bbox_to_anchor=(0., 1.02, 1., .102),
					loc=3, ncol=len(legend_), mode="expand", borderaxespad=0.)
						elif save_number<=10:	# stack values
							var=pd.concat([var,Data])
						else :	# delete old value and add new ones
							var=pd.concat([var[np.shape(Data)[0]:],Data])
						for i in range(self.nbr_graphs):	# update lines
							li[i].set_xdata(var[self.args[i][0]])
							li[i].set_ydata(var[self.args[i][1]])
					ax.relim()
					ax.autoscale_view(True,True,True)
					fig.canvas.draw() 
					if save_number <=10 :
						save_number+=1
						
			if self.mode=="static":
				plt.ion()
				fig=plt.figure()
				ax=fig.add_subplot(111)
				first_round=True
				k=[0]*self.nbr_graphs	# internal value for downsampling
				while True :
					Data=self.inputs[0].recv()	# recv data
					legend_=Data.columns[1:]
					if first_round:	# init at first round
						for i in range(self.nbr_graphs):
							if i==0:
								li=ax.plot(
									Data[self.args[i][0]],Data[self.args[i][1]],
									label='line '+str(i))
							else:
								li.extend(ax.plot(
									Data[self.args[i][0]],Data[self.args[i][1]],
									label='line '+str(i)))
						plt.legend(legend_,bbox_to_anchor=(0., 1.02, 1., .102),
							loc=3,ncol=len(legend_), mode="expand",
							borderaxespad=0.)
						plt.grid()
						fig.canvas.draw()
						first_round=False
					else:	# not first round anymore
						for i in range(self.nbr_graphs):
							data_x=li[i].get_xdata()
							data_y=li[i].get_ydata()
							if len(data_x)>=20000:
								# if more than 20000 values, cut half
								k[i]+=1
								li[i].set_xdata(np.hstack((data_x[::2],
									Data[self.args[i][0]][::2**k[i]])))
								li[i].set_ydata(np.hstack((data_y[::2],
									Data[self.args[i][1]][::2**k[i]])))
							else:
								li[i].set_xdata(np.hstack((data_x,
									Data[self.args[i][0]][::2**k[i]])))
								li[i].set_ydata(np.hstack((data_y,
									Data[self.args[i][1]][::2**k[i]])))
					ax.relim()
					ax.autoscale_view(True,True,True)
					fig.canvas.draw() 
		except Exception as e:
			plt.close('all')
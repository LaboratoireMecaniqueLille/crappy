# testing the speedup of a reshaping

import numpy as np
import time

def refresh_mat():
	matrix_1=np.random.rand(100,100)
	matrix_2=np.random.rand(1000,1000)
	matrix_3=np.random.rand(5000,5000)

	return [matrix_1,matrix_2,matrix_3]

print "double loop"
matrix= refresh_mat()
for i,mat in enumerate(matrix):
	t0=time.time()
	for x in range (np.shape(mat)[0]):
		for y in range (np.shape(mat)[1]):
			mat[x,y]=mat[x,y]*118+163
	t1=time.time()
	print "time for matrix %i : %f" %(i,t1-t0)
	
#time for matrix 0 : 0.014733
#time for matrix 1 : 1.339415
#time for matrix 2 : 33.464568
print "double loop, inverted the 2 variables"
matrix= refresh_mat()
for i,mat in enumerate(matrix):
	t0=time.time()
	for y in range (np.shape(mat)[1]):
		for x in range (np.shape(mat)[0]):
			mat[x,y]=mat[x,y]*118+163
	t1=time.time()
	print "time for matrix %i : %f" %(i,t1-t0)
	
#time for matrix 0 : 0.026583
#time for matrix 1 : 1.343173
#time for matrix 2 : 33.358050
print "simple loop with flattened array"
matrix= refresh_mat()
for i,mat in enumerate(matrix):
	t0=time.time()
	a,b=np.shape(mat)
	#mat.flatten()
	#for x in range (len(mat)):
		#mat[x]=mat[x]*118+163
	mat=mat*118+163
	#np.reshape(mat,(a,b))
	t1=time.time()
	print "time for matrix %i : %f" %(i,t1-t0)
	
#time for matrix 0 : 0.000444
#time for matrix 1 : 0.007575
#time for matrix 2 : 0.105985
### speed up 300x on matrix_2 !!

print "simple loop with flattened array in the other axis"
matrix= refresh_mat()
for i,mat in enumerate(matrix):
	t0=time.time()
	a,b=np.shape(mat)
	np.reshape(mat,(a*b,1))
	print np.shape(mat)
	for x in range (len(mat)):
		mat[x]=mat[x]*118+163
	np.reshape(mat,(a,b))
	t1=time.time()
	print "time for matrix %i : %f" %(i,t1-t0)
#time for matrix 0 : 0.035276
#time for matrix 1 : 3.414348

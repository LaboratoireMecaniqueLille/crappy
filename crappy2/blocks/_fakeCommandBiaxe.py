# coding: utf-8
#from _meta import MasterBlock
#import numpy as np
#import cv2
#import SimpleITK as sitk

#class FakeCommandBiaxe(MasterBlock):
	#"""Receive a signal and translate it for the Biaxe actuator"""
	#def __init__(self, axes=1, spots=4, speed=1):
		#"""
#WIP
		#"""
		#self.axes=axes
		#self.spots=spots
		#self.speed=speed
		#self.last_img=np.zeros((512,512))
		#self.Points_coordinates=np.array([[128,256],[256,384],[384,128],[256,128]])
		#for i in range (4):
			#self.last_img=cv2.circle(self.last_img,(int(self.Points_coordinates[i,1]),int(self.Points_coordinates[i,0])),10,(255,0,0),-1)
	
	#def main(self):
		#try:
			#last_cmd=0
			#while True:
				#Data=self.inputs[0].recv()
				#cmd=Data['signal'].values[0]
				#if cmd!= last_cmd:
					#if cmd > 0: # pulling on axis x
						#self.Points_coordinates[0]-=[self.speed,0]
						#self.Points_coordinates[1]-=[0,self.speed/np.sqrt(2)]
						#self.Points_coordinates[2]+=[self.speed,0]
						#self.Points_coordinates[3]+=[0,self.speed/np.sqrt(2)]
					#elif cmd <0:
						#self.Points_coordinates[0]+=[self.speed,0]
						#self.Points_coordinates[1]+=[0,self.speed/np.sqrt(2)]
						#self.Points_coordinates[2]-=[self.speed,0]
						#self.Points_coordinates[3]-=[0,self.speed/np.sqrt(2)]
					#self.last_img=np.zeros((512,512))
					#for i in range (4):
						#self.last_img=cv2.circle(self.last_img,(int(self.Points_coordinates[i,1]),int(self.Points_coordinates[i,0])),10,(255,0,0),-1)
					#image=sitk.GetImageFromArray(self.last_img)
					#try:
						#sitk.WriteImage(image,os.path.expanduser("~/Bureau/fake_camera_sensor_img.tiff"))
					#except IOError:
						#try:
							#sitk.WriteImage(image,os.path.expanduser("~/Bureau/fake_camera_sensor_img.tiff"))
						#except IOError:
							#raise Exception("Path not found")
					#last_cmd=cmd
		#except (Exception,KeyboardInterrupt) as e:
			#print "Exception in measureComediByStep : ", e
			#raise
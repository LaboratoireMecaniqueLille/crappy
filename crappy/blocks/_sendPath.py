# -*- coding:utf-8 -*-
import Tix
from Tkinter import *
import os
import time
import tkFont
import tkFileDialog
from _meta import MasterBlock
import Tkinter
import tkMessageBox
import threading
import math

class InterfaceSendPath(Frame, MasterBlock):
    def __init__(self, root):
	MasterBlock.__init__(self)
	try:
	    Frame.__init__(self, root, width=1000, height=1000)
	    self.root = root
	    self.t0 = 0
	    self.filepath = "/home/tribo/save_dir/openlog.txt"
	    
	    frameSave = Frame(self.root, width=400, borderwidth=2,relief=GROOVE) 
	    
		    #### DATA RECORD DECLARATION ####
	    #DataRecording
	    self.pathSelectButton = Button(frameSave, text="Browse...", command=self.askdirectory)
	    
	    #self.StartRecordDataButton = Button(frameSave, text="StartRecordData", command=self.go)
	    #self.StopRecordDataButton = Button(frameSave, text="StopRecordData", command=self.stop) 
	    self.filepathVar = Tix.StringVar()
	    self.dirEntry = Entry(frameSave, textvariable=self.filepathVar,width=55)
	    self.filepathVar.set(self.filepath)
	    
		    # defining options for opening a directory
	    self.dir_opt = options = {}
	    options['mustexist'] = True
	    options['parent'] = self.root

	    ####
	    frameSave.grid(row=2, column=1,sticky= "w", padx=10,pady=10)
	    #Inside FrameSave
	    self.pathSelectButton.grid(row=1, column=0, sticky= "w",padx=10,pady=10)
	    self.dirEntry.grid(row=1, column=1, sticky= "w",padx=10,pady=10)
	except Exception as e:
		print e
            
    def main(self):
        self.mainloop()
        self.outputs[0].send(0)
        
        
    def askdirectory(self):
	"""
	Returns a selected directoryname.
	"""
	self.filepath = tkFileDialog.asksaveasfilename()
	self.filepathVar.set(self.filepath)
	self.outputs[0].send(self.filepath)
	self.outputs[0].send(self.filepath)
	

    def pathSelect(self):
	d = Tix.DirSelectBox(master=self.root, command=self.print_selected)
	d.popup()
    
    def getInfo(self):
	return self.filepath
#!/usr/bin/env python

# Luis Enrique Coronado Zuniga

# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.


import Tkinter as tk
import ttk
from Tkinter import StringVar
import tkFileDialog
from ttk import Frame, Style
from tkMessageBox import*
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from numpy import*
from numpy.linalg import*
from scipy import interpolate
import os



class preGUI(Frame):
    """Interfaz for preprocessing data from wearami"""
    def __init__(self, master):
        self.master = master
        """tkinter variable"""
        
        #Set GUI
        self.initUI()
        
        #For change betweem formats
        """if == 0 then i used the new format, new format: a;date;x;y;z, else use direct format: x,y,z"""
        self.option = 1
        
        #list of acceleration files
        self.lista = []

    def setOffset(self,val):
        """Set the offset of each sample"""
        self.plotData("one",0,self.actual_sample,val)
        self.offsets[0,self.actual_sample] = val
        print self.offsets

    def setLimit_r(self,val):
        """Set right limit in all the data"""
        self.plotData("all")
        a = self.figure.add_subplot(111)
        self.limit_r = val
        a.axvline(x=self.limit_l, linewidth=2, color = 'b')
        a.axvline(x=self.limit_r, linewidth=2, color = 'k')
        self.canvas.draw()
    def setLimit_l(self,val):
        """Set left limit in all the data"""
        self.plotData("all")
        a = self.figure.add_subplot(111)
        self.limit_l = val
        a.axvline(x=self.limit_l, linewidth=2, color = 'b')
        a.axvline(x=self.limit_r, linewidth=2, color = 'k')
        self.canvas.draw()



    def initUI(self):
        """Definition of the GUI"""
        master = self.master
        self.master.title("Pre-processing data")
        self.openButton = tk.Button(master, text = 'Select Data', width = 30 ,height = 3, command = self.getData).grid(row=1,column=0, columnspan = 2, padx = 10, pady = 10)


        self.labelname1 = tk.Label(master, text='Change sample')
        self.labelname1.grid(row=3,column=0)
        self.preButton = tk.Button(master, text = 'previous', width = 10 , height = 3, command = self.previous_sample).grid(row=4,column=0, padx = 10, pady = 10)
        self.nextButton = tk.Button(master, text = 'next', width = 10 , height = 3, command = self.next_sample).grid(row=4,column=1, padx = 10, pady = 10)


        self.XButton = tk.Button(master, text = 'x', width = 5 , height = 3, command = self.x_axis).grid(row=6,column=2, padx = 10, pady = 10)
        self.YButton = tk.Button(master, text = 'y', width = 5 , height = 3, command = self.y_axis).grid(row=6,column=3, padx = 10, pady = 10)
        self.ZButton = tk.Button(master, text = 'z', width = 5 , height = 3, command = self.z_axis).grid(row=6,column=4, padx = 10, pady = 10)



        self.saveButton = tk.Button(master, text = 'Save data', width = 20 , height = 3, command = self.save).grid(row=6,column=0, columnspan=2,  padx = 10, pady = 10)


        self.figure = Figure(figsize=(4,3), dpi=100)
        #Figure variable to plot with matplotlib"""
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        # pack and grid dont get along in Mac OS
        if (os == 'Windows'):
            self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        self.canvas.get_tk_widget().grid(row=1,column=2,rowspan = 5, columnspan = 3 ,padx = 10, pady = 10)
        if (os == 'Windows'):
            self.toolbar.grid(row=6,column=1)
        self.canvas.draw()

        #Sample to show in the figure
        self.actual_sample = 1

        #Axis to show in the figure
        self.actual_axis = 0

    #Quit


    def see(self):
        """Plot all the data together"""
        self.plotData("all")

    def next_sample(self):
        """Select a next sample"""
        if(self.actual_sample < len(self.names)-1):
            self.actual_sample = self.actual_sample +1
            #Plot the n sample vs the 0 sample
            self.plotData("one",0,self.actual_sample,self.offsets[0,self.actual_sample])
            #Set offset
            self.off.set(self.offsets[0,self.actual_sample])

    def previous_sample(self):
        """Select a previuos sample"""
        if(self.actual_sample > 1):
            self.actual_sample = self.actual_sample -1
            #Plot the n sample vs the 0 sample
            self.plotData("one",0,int(self.actual_sample),self.offsets[0,self.actual_sample])
            #Set offset
            self.off.set(self.offsets[0,self.actual_sample])
        
    def x_axis(self):
        """Select X axis"""
        self.actual_axis = 0
        self.plotData("one",0,int(self.actual_sample),self.offsets[0,self.actual_sample])

    def y_axis(self):
        """Select Y axis"""
        self.actual_axis = 1
        self.plotData("one",0,int(self.actual_sample),self.offsets[0,self.actual_sample])

    def z_axis(self):
        """Select Z axis"""
        self.actual_axis = 2
        self.plotData("one",0,int(self.actual_sample),self.offsets[0,self.actual_sample])

        
        
    def getData(self):
        """Read the data from txt files"""
        try:
            import matplotlib.pyplot as plt
            self.names = tkFileDialog.askopenfilenames(title='Choose acceleration files')
            self.num = 0
            for name in self.names:
                if(self.option == 0):
                    self.data = genfromtxt(name, delimiter=';')
                else:
                    self.data = genfromtxt(name, delimiter=',')
                self.lista.append(self.data)
                self.num = self.num + 1

            #Offset of each sample
            self.offsets =  zeros((1,self.num));
            self.limit_l = 0
            self.limit_r = 100
            self.off = tk.Scale(self.master, from_=-1000, to=1000, orient=tk.HORIZONTAL, resolution=1, length=600, command = self.setOffset, label = 'Offset')
            self.off.grid(row=7,column=0, columnspan = 5)
            self.ri = tk.Scale(self.master, from_=-800, to=800, orient=tk.HORIZONTAL, resolution=1, length=600, command = self.setLimit_r, label = 'Right limit')
            self.ri.set(self.limit_r)
            self.ri.grid(row=8,column=0, columnspan = 5)
            self.le = tk.Scale(self.master, from_=-1000, to=1000, orient=tk.HORIZONTAL, resolution=1, length=600, command = self.setLimit_l, label = 'Left limt')
            self.le.set(self.limit_l)
            self.le.grid(row=9,column=0, columnspan = 5)
            self.plotData("all")
            self.new_lista = self.lista
        except:
            showerror("Error in the files")


    def save(self):
        """Save the data """
        filename = tkFileDialog.asksaveasfilename()
        print "Saving data ... \n"
        for i in range(self.num):
            lmI = int(self.limit_l)
            lmR = int(self.limit_r)
            samples = int(self.limit_r) - int(self.limit_l)
            offset = int(self.offsets[0,i])
            init = lmI - offset
            n,m = self.lista[i].shape
            
            if lmI < offset:
                print "Error in saving data, sample", i, "increase left limit"
            elif n + offset <  lmR:
                print "Error in saving data, sample", i, "decrease rigth limit"
            else:
                print "Saving data", i, "OK"
                if(self.option == 0):
                    savetxt(filename + "(" + str(i+1) +').txt', self.lista[i][init:init+samples,2:] , delimiter=' ', fmt='%f')
                else:
                    savetxt(filename + "(" + str(i+1) +').txt', self.lista[i][init:init+samples,:] , delimiter=' ', fmt='%f')
                

    def plotData(self, show = "one", ref_sample = 0, n_sample = 1, offset = 0):
        """Plot the data
        Keyword arguments:
        show -- "one" to see only the data of one sample, "all" to see all the samples  (default "one")
        ref_sample -- reference sample (default 0)
        n_sample -- number of sample to compare with the reference sample (default 1)
        offset -- offset of the sample (default 0)
        """
        if(self.option == 0):
            number = self.actual_axis +1
        else:
            number = self.actual_axis 
            
        if(show == 'all'):
            a = self.figure.add_subplot(111)
            a.clear()
            for i in range(self.num):
                n,m = self.lista[i].shape
                x = arange(self.offsets[0,i],n+self.offsets[0,i],1)
                a.plot(x, self.lista[i].transpose()[number ])
                a.set_ylabel("Position $(rad)$")
                a.grid(True)
                self.canvas.draw()
        if(show == 'one'):
            a = self.figure.add_subplot(111)
            a.clear()

            offset = int(offset)
            n,m = self.lista[ref_sample].shape
            x = arange(0,n,1)
            a.plot(x, self.lista[ref_sample].transpose()[number ])
            print offset

            temp = self.lista[n_sample]
            n,m = temp.shape
            x = arange(offset,n+offset,1)
            a.plot(x, temp.transpose()[number ])


            a.grid(True)
            self.canvas.draw()

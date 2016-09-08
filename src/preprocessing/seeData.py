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



class seeData(Frame):
    """GUI to se a set of data after preprocessing"""
    def __init__(self, master):
        self.master = master
        """tkinter variable"""
        self.initUI()
        self.lista = []



    def initUI(self):
        """Definition of the GUI"""
        master = self.master
        self.master.title("See data")
        self.openButton = tk.Button(master, text = 'Select Data', width = 30 , height = 3, command = self.getData).grid(row=1,column=0, padx = 10, pady = 10)
        self.seeallButton = tk.Button(master, text = 'Plot data', width = 30 , height = 3, command = self.see).grid(row=2,column=0, padx = 10, pady = 10)
 
        self.labelname1 = tk.Label(master, text='Axis')
        self.labelname1.grid(row=3,column=0)
        self.numberS = StringVar()
        self.numbertxt = tk.Entry(master, width = 20, textvariable = self.numberS )
        self.numbertxt.insert(0, "1")
        self.numbertxt.grid(row=4,column=0)

        self.figure = Figure(figsize=(4,3), dpi=100)
        #Figure variable to plot with matplotlib"""
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        if (os == 'Windows'):
            self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        self.canvas.get_tk_widget().grid(row=1,column=1,rowspan = 5,padx = 10, pady = 10)
        if (os == 'Windows'):
            self.toolbar.grid(row=6,column=1)
        self.canvas.draw()


    def see(self):
        """Plot all the data together"""
        self.plotData();

    def getData(self):
        """Read the data from txt files"""
        try:
            import matplotlib.pyplot as plt
            self.names = tkFileDialog.askopenfilenames(title='Choose acceleration files')
            self.num = 0
            for name in self.names:
                self.data = genfromtxt(name, delimiter=' ')
                self.lista.append(self.data)
                self.num = self.num + 1

        except:
            showerror("Error in the files")

    def plotData(self):
        """Plot the data"""
        number = int(self.numberS.get()) -1
        a1 = self.figure.add_subplot(311)
        a2 = self.figure.add_subplot(312)
        a3 = self.figure.add_subplot(313)
        a1.clear()
        a2.clear()
        a3.clear()
        for i in range(self.num):
            n,m = self.lista[i].shape
            x = arange(0,n,1)
            a1.plot(x, self.lista[i].transpose()[0])
            a2.plot(x, self.lista[i].transpose()[1])
            a3.plot(x, self.lista[i].transpose()[2])

        a1.grid(True)
        a2.grid(True)
        a3.grid(True)
        self.canvas.draw()

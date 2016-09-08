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
import preGUI
import seeData


class MainGUI:
    """Main windows definition"""
    
    def __init__(self, master):
        """Define main window"""
        self.master = master
        self.master.title("EMARO+, Genoa Universisty, Preprocessing data")
        self.openButton = tk.Button(master, text = 'Manual pre-processing of inertial data', width = 50, command = self.pre_window)
        self.openButton.grid(row=2,column=0, padx = 2, pady = 2)
        self.openButton2 = tk.Button(master, text = 'See preprocessing data', width = 50, command = self.see_window)
        self.openButton2.grid(row=3,column=0, padx = 2, pady = 2)

    def pre_window(self):
        """Open Window for manual preprocessing"""
        self.newWindow = tk.Toplevel(self.master)
        self.newWindow.geometry("850x650")
        self.app = preGUI.preGUI(self.newWindow)
        
    def see_window(self):
        """Open Window for see the data after proprocessing"""
        self.newWindow = tk.Toplevel(self.master)
        self.newWindow.geometry("800x400")
        self.app = seeData.seeData(self.newWindow)
        

def main():
    """Main fuction"""
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()

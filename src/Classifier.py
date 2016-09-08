#!/usr/bin/env python

"""@See preprocessed data
"""
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy import*
from numpy.linalg import*
from scipy import interpolate
from scipy.signal import filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import filter_design as ifd
from scipy.stats import multivariate_normal
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.mixture import GMM
from GestureModel import*
import time


class Classifier:
    """Class used to classify a gesture"""

    def __init__(self):
        print "New classifier"
        self.dataIndex = 0

        #APPLY IIR FILTER TO GET THE GRAVITY COMPONENTS
        #IIR filter parameters (all frequencies are in Hz)
        Fs = 32;            # sampling frequency
        Fpass = 0.25;       # passband frequency
        Fstop = 2;          # stopband frequency
        Apass = 0.001;      # passband ripple (dB)
        Astop = 100;        # stopband attenuation (dB)
        match = 'pass';     # band to match exactly
        #Create the IIR filter


        # iirdesign agruements
        Wip = (Fpass)/(Fs/2)
        Wis = (Fstop+1e6)/(Fs/2)
        Rp = Apass             # passband ripple
        As = Astop            # stopband attenuation

        # The iirdesign takes passband, stopband, passband ripple, 
        # and stop attenuation. 
        self.bb, self.ab = ifd.iirdesign(Wip, Wis, Rp, As, ftype='cheby1')


    def classify(self,model):
        """Set the model to classify

        :param model: a instance of a GestureModel class
        """
        self.model = model
        self.window_size,m = self.model.component["gravity"][0].shape
        self.window_size = self.window_size + 64
        print "window size", self.window_size
        self.window = zeros((self.window_size,3))


    def calculateW(self,sfiles,model):
        """This function can be used to obtain a better value of the weights

            :param sfiles: list of files
            :model model: model to be used 
        """

        window_size,m = (model.component["gravity"][0].shape)
        print "models size = ", window_size
        window = zeros((window_size,3))

        totalG = 0
        totalB = 0

        k = 0
        for file in sfiles:
            numWritten = 0
            data = genfromtxt(file, delimiter=' ')
            numSamples,m = data.shape

            distanceG = 0
            distanceB = 0
            
            for j in range (numSamples):
                one_sample = data[j]
                window,numWritten = self.createWindow(one_sample, window, window_size, numWritten)
                if (numWritten >= window_size):
                    # Compute the acceleration components of the current window of samples
                    gravity,body = self.analyzeActualWindow(window,window_size,delay = 0)
                    distanceG = distanceG + self.compareComponent(model,"gravity",gravity,window_size, "mh")
                    distanceB = distanceB + self.compareComponent(model,"body",body,window_size,"mh")
                    k = k + 1

        #New implemntation
            totalG = totalG + distanceG
            totalB = totalB + distanceB

        mean_g =  totalG/k
        mean_b = totalB/k

        g_Weight = mean_g/(mean_g + mean_b)
        b_Weight = w_b = mean_b/(mean_g + mean_b)

        print "Weight of the gravity", g_Weight
        print "Weight of the body", b_Weight

        return g_Weight, b_Weight

    def validate_from_file(self,sfile,delim = ' '):
        """
        This function returns a probability vector given a .txt file with data information

        :param sfile: path and name of the file to be validated
        :param delim: delimiter of the data in the txt file
        :return possibilities_vector: vector of possibilities which can be used for recognize a gesture
        """
        numWritten = 0
        data = genfromtxt(sfile, delimiter=delim)
        numSamples,m = data.shape
        print "Num samples", numSamples
        
        possibilities_vector = zeros((numSamples,1))
        for i in range (numSamples):
            one_sample = data[i]

            t0 = time.clock()
            self.window,numWritten = self.createWindow(one_sample, self.window, self.window_size, numWritten)

                
            if (numWritten >= self.window_size):
                # Compute the acceleration components of the current window of samples
                gravity,body = self.analyzeActualWindow(self.window,self.window_size,delay = 64);

                possibilities_vector[i,0] = self.compareModel(self.model,gravity, body,self.window_size);

                tm = time.clock() - t0
                #print "comparar =", tm
            
        return possibilities_vector

    def online_validation(self,x,y,z):
        """This function is used for online classification from a sample of sensory data

            :param x: x value
            :param y: y value
            :param z: z value
        """
        one_sample = array([x,y,z])
        import time
        t0 = time.clock()
        self.window,self.dataIndex = self.createWindow(one_sample, self.window, self.window_size,self.dataIndex)

        if (self.dataIndex >= self.window_size):
                 
            gravity,body = self.analyzeActualWindow(self.window,self.window_size,delay = 64);
            possibilities = self.compareModel(self.model,gravity, body,self.window_size)

            tm = time.clock() - t0
            
            return possibilities
        else:
            return 0
                    
    
    def createWindow(self,one_sample, window, window_size, numWritten):
        """
        This function creates a windows used to compare with the models

        :param one_sample: vector of sensory data in time t
        :param windows: old windows
        :param windows_size: size that can have the windows
        :param numWritten: count used to itearte betweem samples
        """
        #Fill window if numWritten < window_size
        if numWritten < window_size:
            for j in range(3):
                window[numWritten,j] = one_sample[j];
            numWritten = numWritten + 1
        #shift window and update
        else:
            for i in range(window_size-1):
                for j in range(3):
                        window[i,j] = window[i+1,j];
            for j in range(3):
                window[window_size - 1,j] = one_sample[j];
            numWritten = numWritten + 1
        return window,numWritten

    def analyzeActualWindow(self,window,numSamples,delay = 0):
        """ AnalyzeActualWindow separates the gravity and body acceleration features
            contained in the window of real-time acceleration data, by first reducing
            the noise on the raw data with a median filter and then discriminating
            between the features with a low-pass IIR filter.

            :param window: windows to be processed and analalized
            :param numSamples: is equal to the windows size
            :param delay: delay of samples
            """
        
        self.delay = delay
        #REDUCE THE NOISE ON THE SIGNALS BY MEDIAN FILTERING
        n = 3  #order of the median filter
        x_axis = medfilt(window[:,0],n)
        y_axis = medfilt(window[:,1],n)
        z_axis = medfilt(window[:,2],n)


        #Gravity components
        g1 = lfilter(self.bb,self.ab,x_axis)
        g2 = lfilter(self.bb,self.ab,y_axis)
        g3 = lfilter(self.bb,self.ab,z_axis)

        #m, = x_axis.shape
        #import matplotlib.pyplot as plt
        #x = arange(0,m,1)
        #plt.plot(x, x_axis)
        #plt.plot(x, g1)
        #plt.show()


        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION  (PREGUNTA)
        gravity = zeros((numSamples-delay,3));
        body = zeros((numSamples-delay,3));

        i=delay
        while(i < numSamples):
            gravity[i-delay,0] = g1[i];
            gravity[i-delay,1] = g2[i];
            gravity[i-delay,2] = g3[i];
            body[i-delay,0] = x_axis[i-delay] - gravity[i-delay,0];
            body[i-delay,1] = y_axis[i-delay] - gravity[i-delay,1];
            body[i-delay,2] = z_axis[i-delay] - gravity[i-delay,2];
            i = i + 1

        #m, = x_axis.shape
        #import matplotlib.pyplot as plt
        #x = arange(0,m-delay,1)
        #plt.plot(x, gravity[:,0])
        #plt.plot(x, body[:,0])

        #plt.show()

        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION
        return gravity, body

    def compareComponent(self,model,string_component,component,window_size, distance_type = "mh"):
        """Returns the distance from the model

            :param model: model to be compared
            :param string_component: string that indicate the component to be comparated (gravity or body)
            :param window_size: size of a windows to be compared
            :param distanc_type: metrics of the distance, by defual is the Mahalanobis distance "mh"
        """
        mean =  model.component[string_component][0] #0 is the mean
        sigma =  model.component[string_component][1] #1 is sigma

        
        dist = zeros((1,window_size-self.delay));
        if(distance_type == "mh"): #Mahalanobis distance
            for i in range(window_size-self.delay):

                rest = mean[i]-component[i]
                restT = transpose(rest)

                inverse_sigma = linalg.inv(sigma[i*3:((i+1)*3)])
                inverse_sigma = dot(inverse_sigma,restT)
                dist[0,i] =  dot(rest,inverse_sigma)
                
        import numpy as np
        distance = np.mean(dist) #mean
        return distance



    def compareModel(self,model,gravity,body,window_size):
        """
        Compare model with a window of data sensory

        :param model: model to be compared
        :param gravity: gravity component of the window
        :param body: body component of the window
        :param window_size: integer that indicate the size of the window
        """
        distanceG = self.compareComponent(model,"gravity",gravity,window_size, "mh")
        distanceB = self.compareComponent(model,"body",body,window_size,"mh")
        overall = model.Weights["gravity"]*distanceG + model.Weights["body"]*distanceB

        """ compute the possibility value of each model
        (mapping of the likelihoods from [0..threshold(i)] to [1..0]"""
        possibilities = 1 - overall/model.threashold;
        if (possibilities < 0):
            possibilities = 0
        return possibilities
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()              
                    
